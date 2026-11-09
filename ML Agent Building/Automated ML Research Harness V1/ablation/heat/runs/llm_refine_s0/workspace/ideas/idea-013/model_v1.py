import numpy as np
import torch
import torch.nn as nn

class NoisyEnsembleMLP:
    def __init__(self, params, seed):
        self.p = params
        self.seed = seed
        self.models = []
    def _net(self, d_in, d_out, h):
        return nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, d_out))
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        n = len(X)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n)
        nv = max(1, int(0.1 * n))
        vi, ti = idx[:nv], idx[nv:]
        Xt, yt = torch.tensor(X[ti]), torch.tensor(y[ti])
        Xv, yv = torch.tensor(X[vi]), torch.tensor(y[vi])
        h = int(self.p.get('hidden', 256)); E = int(self.p.get('epochs', 150)); lr = float(self.p.get('lr', 2e-3)); sig = float(self.p.get('noise', 0.05))
        self.models = []
        for m in range(int(self.p.get('n_models', 3))):
            torch.manual_seed(self.seed * 100 + m)
            net = self._net(X.shape[1], y.shape[1], h)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
            best, best_state, bad = 1e18, None, 0
            for ep in range(E):
                net.train()
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 128):
                    b = perm[i:i+128]
                    xb = Xt[b] + sig * torch.randn_like(Xt[b])
                    loss = ((net(xb) - yt[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    vl = ((net(Xv) - yv) ** 2).mean().item()
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 25:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.models.append(net)
        self._1d = (np.asarray(y).shape[1] == 1)
        return self
    def predict(self, X):
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            P = np.mean([m(X).numpy() for m in self.models], axis=0)
        return P[:, 0] if self._1d else P
    def get_params(self, deep=True):
        return {}

def build_model(params: dict, task: str, seed: int):
    torch.set_num_threads(1)
    return NoisyEnsembleMLP(params, seed)
