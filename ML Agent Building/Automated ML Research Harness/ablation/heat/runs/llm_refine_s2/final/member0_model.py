import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

class NoisyEnsembleMLP(BaseEstimator, RegressorMixin):
    def __init__(self, n_models=3, hidden=256, epochs=150, lr=2e-3, wd=1e-4, noise=0.05, seed=0):
        self.n_models = n_models; self.hidden = hidden; self.epochs = epochs
        self.lr = lr; self.wd = wd; self.noise = noise; self.seed = seed

    def _net(self, d_in, d_out):
        h = self.hidden
        return nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.SiLU(),
                             nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(),
                             nn.Linear(h, d_out))

    def fit(self, X, y):
        torch.set_num_threads(1)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        self._1d = (np.asarray(y).shape[1] == 1)
        Y = torch.tensor(y, dtype=torch.float32)
        n = X.shape[0]
        n_val = max(1, int(0.1 * n))
        self.models_ = []
        for m in range(self.n_models):
            g = torch.Generator().manual_seed(self.seed * 100 + m)
            torch.manual_seed(self.seed * 100 + m)
            perm = torch.randperm(n, generator=g)
            vi, ti = perm[:n_val], perm[n_val:]
            net = self._net(X.shape[1], Y.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            best, best_state, bad = float('inf'), None, 0
            for ep in range(self.epochs):
                net.train()
                p = ti[torch.randperm(len(ti), generator=g)]
                for i in range(0, len(p), 128):
                    b = p[i:i+128]
                    xb = X[b] + self.noise * torch.randn(len(b), X.shape[1], generator=g)
                    loss = ((net(xb) - Y[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    vl = ((net(X[vi]) - Y[vi]) ** 2).mean().item()
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 25:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.models_.append(net)
        return self

    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            P = torch.stack([m(X) for m in self.models_]).mean(0).numpy()
        return P[:, 0] if self._1d else P

def build_model(params: dict, task: str, seed: int):
    return NoisyEnsembleMLP(n_models=int(params.get('n_models', 3)), hidden=int(params.get('hidden', 256)),
                            epochs=int(params.get('epochs', 150)), lr=float(params.get('lr', 2e-3)),
                            wd=float(params.get('wd', 1e-4)), noise=float(params.get('noise', 0.05)), seed=seed)
