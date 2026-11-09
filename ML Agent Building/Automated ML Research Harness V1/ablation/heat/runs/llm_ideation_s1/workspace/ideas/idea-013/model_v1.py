import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

class RidgeResidualMLP(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, hidden=256, epochs=200, lr=2e-3, wd=1e-4, n_models=3, seed=0):
        self.alpha = alpha; self.hidden = hidden; self.epochs = epochs
        self.lr = lr; self.wd = wd; self.n_models = n_models; self.seed = seed

    def _net(self, d_in, d_out):
        h = int(self.hidden)
        return nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, d_out))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1: y = y[:, None]
        self.ridge_ = Ridge(alpha=self.alpha).fit(X, y)
        r = (y - self.ridge_.predict(X)).astype(np.float32)
        n = len(X); rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n); nv = max(1, int(0.1 * n))
        vi, ti = idx[:nv], idx[nv:]
        Xt = torch.tensor(X); rt = torch.tensor(r)
        self.nets_ = []
        for m in range(int(self.n_models)):
            torch.manual_seed(self.seed * 100 + m)
            net = self._net(X.shape[1], r.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            E = int(self.epochs)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
            best, best_state, patience = 1e18, None, 0
            for ep in range(E):
                net.train()
                perm = ti[np.random.RandomState(ep + 1000 * m).permutation(len(ti))]
                for s in range(0, len(perm), 128):
                    b = perm[s:s + 128]
                    opt.zero_grad()
                    loss = ((net(Xt[b]) - rt[b]) ** 2).mean()
                    loss.backward(); opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    vl = ((net(Xt[vi]) - rt[vi]) ** 2).mean().item()
                if vl < best - 1e-6:
                    best = vl; best_state = {k: v.clone() for k, v in net.state_dict().items()}; patience = 0
                else:
                    patience += 1
                    if patience > 25: break
            net.load_state_dict(best_state)
            self.nets_.append(net)
        self.single_ = (y.shape[1] == 1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        base = self.ridge_.predict(X)
        if base.ndim == 1: base = base[:, None]
        Xt = torch.tensor(X)
        with torch.no_grad():
            corr = np.mean([n.eval()(Xt).numpy() for n in self.nets_], axis=0)
        out = base + corr
        return out[:, 0] if self.single_ else out

def build_model(params, task, seed):
    torch.set_num_threads(1)
    return RidgeResidualMLP(alpha=params.get('alpha', 1.0), hidden=params.get('hidden', 256), epochs=params.get('epochs', 200), lr=params.get('lr', 2e-3), wd=params.get('wd', 1e-4), n_models=params.get('n_models', 3), seed=seed)
