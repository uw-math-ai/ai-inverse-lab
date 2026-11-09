"""Batched path-wise Monte Carlo in torch (CUDA or CPU).

Extends the notebook's batched sampler:
  * float64 cumulative probabilities over the augmented states, including the cemetery;
  * many graphs x |B|^2 pairs x m walks per launch, using per-graph sparse tables
    (G, n_max+1, D) of reachable states and cdfs. Padded vertices get a self-loop, and all
    cemeteries are remapped to the shared index n_max;
  * small integer dtypes for states (int16 when n_max < 32k);
  * separate counts for hits at each t, killed, and censored; nothing is dropped;
  * finished walks are compacted away each step, with an early exit when none are left.

Every item draws its uniforms from its own torch.Generator. Walks are kept contiguous by
item, and boolean compaction keeps their order, so each item's draws depend only on its own
seed. Results are therefore the same however items are packed into launches.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .mc_common import MCItem, bytes_per_walk, check_item_counts

log = logging.getLogger("fptsim")


class TorchMC:
    name = "torch"

    def __init__(self, device="cpu", max_batch_walks=50_000_000, cpu_batch_walks=4_000_000,
                 mem_fraction=0.5, max_items=4096):
        self.device = torch.device(device)
        self.max_batch_walks = int(max_batch_walks)
        self.cpu_batch_walks = int(cpu_batch_walks)
        self.mem_fraction = mem_fraction
        self.max_items = max_items
        self.walk_steps = 0

    # ------------------------------------------------------------ memory-aware packing
    def _limits(self) -> tuple[int, float]:
        """(max walks per launch, byte budget per launch)."""
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            return self.max_batch_walks, self.mem_fraction * free
        return min(self.max_batch_walks, self.cpu_batch_walks), float("inf")

    def run_items(self, items: list[MCItem], stop=None):
        """Yield (index, counts (k, k, buckets, T+2), seconds) as launches complete."""
        i = 0
        while i < len(items):
            if stop is not None and stop():
                return
            cap, budget = self._limits()
            group, total, D = [i], items[i].walks, items[i].D
            j = i + 1
            while j < len(items) and len(group) < self.max_items:
                D2, w = max(D, items[j].D), total + items[j].walks
                if w > cap or w * bytes_per_walk(D2) > budget:
                    break
                group.append(j)
                total, D = w, D2
                j += 1
            if total * bytes_per_walk(D) > budget:
                log.warning("one Monte Carlo item needs %.2f GiB; lower simulation.chunk_walks",
                            total * bytes_per_walk(D) / 2 ** 30)
            started = time.perf_counter()
            results = self._launch([items[x] for x in group])
            seconds = time.perf_counter() - started
            for x, counts in zip(group, results):
                yield x, counts, seconds * items[x].walks / total
            i = j

    # ------------------------------------------------------------ one launch
    @torch.no_grad()
    def _launch(self, items: list[MCItem]) -> list[np.ndarray]:
        dev = self.device
        T = items[0].T
        cps = items[0].checkpoints
        if any(it.T != T or not np.array_equal(it.checkpoints, cps) for it in items):
            raise ValueError("items in one launch must share T and checkpoints")
        L, nb, nbins = len(items), len(cps), T + 2
        n1 = max(it.n for it in items) + 1
        cem = n1 - 1                                   # shared cemetery index
        D = max(it.D for it in items)
        sdtype = torch.int16 if n1 < 32_000 else torch.int32

        # stacked per-graph tables; padded rows are self-loops, padded columns repeat the last state
        nbr = np.broadcast_to(np.arange(n1)[None, :, None], (L, n1, D)).copy()
        cdf = np.ones((L, n1, D))
        for l, it in enumerate(items):
            real = np.where(it.nbr[:it.n] == it.n, cem, it.nbr[:it.n])
            nbr[l, :it.n, :it.D] = real
            nbr[l, :it.n, it.D:] = real[:, -1:]
            cdf[l, :it.n, :it.D] = it.cdf[:it.n]
        nbr_t = torch.from_numpy(nbr.reshape(L * n1, D)).to(dev, sdtype)
        cdf_t = torch.from_numpy(cdf.reshape(L * n1, D)).to(dev)
        cps_t = torch.as_tensor(cps, dtype=torch.int64, device=dev)

        # per-pair lookups and per-walk state, walks contiguous by item then pair
        pair_target, pair_base, pair_item, offsets = [], [], [], []
        states, pairs, buckets = [], [], []
        off = 0
        for l, it in enumerate(items):
            k2 = it.k * it.k
            B = np.asarray(it.B, dtype=np.int64)
            src, tgt = np.repeat(B, it.k), np.tile(B, it.k)
            pair_target.append(tgt)
            pair_base.append(np.full(k2, l * n1, dtype=np.int64))
            pair_item.append(np.full(k2, l, dtype=np.int64))
            offsets.append(off)
            states.append(torch.as_tensor(src, device=dev).repeat_interleave(it.m))
            pairs.append((off + torch.arange(k2, device=dev)).repeat_interleave(it.m))
            widx = it.start + torch.arange(it.m, device=dev, dtype=torch.int64)
            buckets.append(torch.searchsorted(cps_t, widx, right=True).repeat(k2))
            off += k2
        n_pairs = off
        pair_target = torch.as_tensor(np.concatenate(pair_target), device=dev).to(sdtype)
        pair_base = torch.as_tensor(np.concatenate(pair_base), device=dev)
        pair_item = torch.as_tensor(np.concatenate(pair_item), device=dev)
        state = torch.cat(states).to(sdtype)
        pair = torch.cat(pairs).to(torch.int32)
        bucket = torch.cat(buckets).to(torch.int16)
        del states, pairs, buckets

        gens = []
        for it in items:
            g = torch.Generator(device=dev)
            g.manual_seed(int(it.seed))
            gens.append(g)

        counts = torch.zeros(n_pairs * nb * nbins, dtype=torch.int64, device=dev)
        for t in range(T):
            alive = state.numel()
            if alive == 0:
                break                                  # every walk has finished
            self.walk_steps += alive
            pl = pair.long()
            seg = torch.bincount(pair_item[pl], minlength=L).tolist()
            us = [torch.rand((c, 1), generator=gens[l], device=dev, dtype=torch.float64)
                  for l, c in enumerate(seg) if c > 0]
            u = us[0] if len(us) == 1 else torch.cat(us)
            rows = pair_base[pl] + state.long()
            col = (u > cdf_t[rows]).sum(dim=1).clamp_(max=D - 1)
            nxt = nbr_t[rows, col]
            hit = nxt == pair_target[pl]
            fin = hit | (nxt == cem)
            if bool(fin.any()):
                b = T - (T - t) * hit[fin].long()      # bin t on a hit, bin T when killed
                idx = (pl[fin] * nb + bucket[fin].long()) * nbins + b
                counts += torch.bincount(idx, minlength=counts.numel())
                keep = ~fin
                state, pair, bucket = nxt[keep], pair[keep], bucket[keep]
            else:
                state = nxt
        if state.numel():
            idx = (pair.long() * nb + bucket.long()) * nbins + (T + 1)
            counts += torch.bincount(idx, minlength=counts.numel())

        counts = counts.view(n_pairs, nb, nbins).cpu().numpy()
        out = []
        for it, o in zip(items, offsets):
            c = counts[o:o + it.k * it.k].reshape(it.k, it.k, nb, nbins).copy()
            check_item_counts(it, c)
            out.append(c)
        return out
