import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class NoisyEnsembleMLP(RegressorMixin, BaseEstimator):
    def __init__(self, params=None, seed=0):
        self.params = params
        self.seed = seed

    def _net(self, d_in, d_out, h):
        return nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, d_out))

    def fit(self, X, y):
        p = dict(self.params or {})
        seed = int(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._1d_ = (y.ndim == 1)
        if y.ndim == 1:
            y = y[:, None]
        n = len(X)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        nv = max(1, int(0.1 * n))
        vi, ti = idx[:nv], idx[nv:]
        if len(ti) == 0:
            ti = idx
        Xt, yt = torch.tensor(X[ti]), torch.tensor(y[ti])
        Xv, yv = torch.tensor(X[vi]), torch.tensor(y[vi])
        h = int(p.get('hidden', 256))
        ep = int(p.get('epochs', 120))
        lr = float(p.get('lr', 2e-3))
        wd = float(p.get('wd', 1e-4))
        sig = float(p.get('noise', 0.05))
        bs = int(p.get('batch_size', 128))
        patience = int(p.get('patience', 20))
        torch.set_num_threads(1)
        models = []
        for m in range(int(p.get('n_models', 3))):
            torch.manual_seed(seed * 100 + m)
            net = self._net(X.shape[1], y.shape[1], h)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, ep))
            best, best_state, bad = float('inf'), None, 0
            for e in range(ep):
                net.train()
                perm = torch.randperm(len(Xt))
                for b in range(0, len(Xt), bs):
                    bi = perm[b:b + bs]
                    xb = Xt[bi] + sig * torch.randn_like(Xt[bi])
                    loss = ((net(xb) - yt[bi]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    vl = ((net(Xv) - yv) ** 2).mean().item()
                if np.isfinite(vl) and vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            models.append(net)
        self.models_ = models
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            P = np.mean([m(X).numpy() for m in self.models_], axis=0)
        if self._1d_:
            P = P[:, 0]
        return P


def build_model(params, task, seed):
    return NoisyEnsembleMLP(params=dict(params or {}), seed=int(seed))
