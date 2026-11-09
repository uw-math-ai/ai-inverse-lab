import numpy as np
import torch
import torch.nn as nn

class _Res(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))
    def forward(self, x):
        return x + self.f(x)

class _Net(nn.Module):
    def __init__(self, d, o, w, b):
        super().__init__()
        self.inp = nn.Linear(d, w)
        self.blocks = nn.Sequential(*[_Res(w) for _ in range(b)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, o))
    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))

class NoisyResEnsemble:
    def __init__(self, width=384, blocks=3, epochs=200, lr=0.002, wd=1e-4, noise=0.05, n_models=3, seed=0):
        self.width = width; self.blocks = blocks; self.epochs = epochs; self.lr = lr
        self.wd = wd; self.noise = noise; self.n_models = n_models; self.seed = seed
    def get_params(self, deep=True):
        return dict(width=self.width, blocks=self.blocks, epochs=self.epochs, lr=self.lr, wd=self.wd, noise=self.noise, n_models=self.n_models, seed=self.seed)
    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self
    def fit(self, X, y):
        torch.set_num_threads(1)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1: y = y[:, None]
        Y = torch.tensor(y)
        n = X.shape[0]
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n); nv = max(1, n // 10)
        vi, ti = perm[:nv], perm[nv:]
        self.models = []
        for m in range(self.n_models):
            torch.manual_seed(self.seed * 100 + m)
            net = _Net(X.shape[1], Y.shape[1], self.width, self.blocks)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * int(np.ceil(len(ti) / 128)))
            best, best_state, bad = 1e9, None, 0
            for ep in range(self.epochs):
                net.train()
                p = ti[np.random.permutation(len(ti))]
                for i in range(0, len(p), 128):
                    b = p[i:i + 128]
                    xb = X[b] + self.noise * torch.randn_like(X[b])
                    loss = ((net(xb) - Y[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                net.eval()
                with torch.no_grad():
                    vl = ((net(X[vi]) - Y[vi]) ** 2).mean().item()
                if vl < best:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad > 30: break
            net.load_state_dict(best_state)
            self.models.append(net)
        self._1d = (np.asarray(y).shape[1] == 1)
        return self
    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            P = np.mean([m(X).numpy() for m in self.models], axis=0)
        return P

def build_model(params: dict, task: str, seed: int):
    return NoisyResEnsemble(seed=seed, **params)
