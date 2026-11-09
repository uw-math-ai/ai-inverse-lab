import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
    def forward(self, x):
        return x + self.net(x)

class Net(nn.Module):
    def __init__(self, d, o, h, b):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.blocks = nn.Sequential(*[Block(h) for _ in range(b)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, o))
    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))

class NoisyResEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=256, blocks=3, noise=0.05, lr=2e-3, epochs=120, n_models=3, wd=1e-4, seed=0):
        self.hidden = hidden; self.blocks = blocks; self.noise = noise; self.lr = lr
        self.epochs = epochs; self.n_models = n_models; self.wd = wd; self.seed = seed
    def fit(self, X, y):
        torch.set_num_threads(1)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = np.asarray(y, dtype=np.float32)
        self._1d = y.ndim == 1
        if self._1d: y = y[:, None]
        Y = torch.tensor(y)
        n = X.shape[0]
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n); nv = max(1, int(0.1 * n))
        vi, ti = perm[:nv], perm[nv:]
        self.models_ = []
        for m in range(self.n_models):
            torch.manual_seed(self.seed * 100 + m)
            net = Net(X.shape[1], Y.shape[1], self.hidden, self.blocks)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * ((len(ti) + 127) // 128))
            best, best_state, bad = 1e9, None, 0
            for ep in range(self.epochs):
                net.train()
                idx = torch.tensor(rng.permutation(ti))
                for s in range(0, len(idx), 128):
                    b = idx[s:s + 128]
                    xb = X[b] + self.noise * torch.randn_like(X[b])
                    loss = ((net(xb) - Y[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                net.eval()
                with torch.no_grad():
                    vl = ((net(X[vi]) - Y[vi]) ** 2).mean().item()
                if vl < best - 1e-6:
                    best = vl; best_state = {k: v.clone() for k, v in net.state_dict().items()}; bad = 0
                else:
                    bad += 1
                    if bad > 20: break
            net.load_state_dict(best_state)
            self.models_.append(net)
        return self
    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            p = np.mean([m(X).numpy() for m in self.models_], axis=0)
        return p[:, 0] if self._1d else p

def build_model(params, task, seed):
    p = dict(params)
    return NoisyResEnsemble(hidden=int(p.get('hidden', 256)), blocks=int(p.get('blocks', 3)), noise=float(p.get('noise', 0.05)), lr=float(p.get('lr', 2e-3)), epochs=int(p.get('epochs', 120)), n_models=int(p.get('n_models', 3)), wd=float(p.get('wd', 1e-4)), seed=seed)
