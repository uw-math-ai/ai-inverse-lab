import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge

class BlendNoisyMLPKRR(BaseEstimator, RegressorMixin):
    def __init__(self, n_models=4, hidden=256, epochs=150, lr=2e-3, wd=1e-4, noise=0.03, krr_alpha=1e-3, gamma_mult=0.3, blend=0.5, seed=0):
        self.n_models = n_models; self.hidden = hidden; self.epochs = epochs
        self.lr = lr; self.wd = wd; self.noise = noise
        self.krr_alpha = krr_alpha; self.gamma_mult = gamma_mult; self.blend = blend; self.seed = seed

    def fit(self, X, y):
        torch.set_num_threads(1)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._1d = (y.ndim == 1)
        if self._1d:
            y = y[:, None]
        n, d = X.shape
        out = y.shape[1]
        var = float(X.var()) if float(X.var()) > 0 else 1.0
        self.krr_ = KernelRidge(alpha=self.krr_alpha, kernel='rbf', gamma=self.gamma_mult / (d * var))
        self.krr_.fit(X, y)
        Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
        h = int(self.hidden); bs = 128
        steps_per_epoch = max(1, int(np.ceil(n / bs)))
        self.nets_ = []
        for m in range(int(self.n_models)):
            torch.manual_seed(self.seed * 100 + m)
            net = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(),
                                nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(),
                                nn.Linear(h, out))
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(self.epochs) * steps_per_epoch)
            lossf = nn.MSELoss()
            net.train()
            for ep in range(int(self.epochs)):
                perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    xb = Xt[idx]
                    if self.noise > 0:
                        xb = xb + self.noise * torch.randn_like(xb)
                    opt.zero_grad()
                    loss = lossf(net(xb), yt[idx])
                    loss.backward()
                    opt.step(); sched.step()
            net.eval()
            self.nets_.append(net)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.from_numpy(X)
        with torch.no_grad():
            p_mlp = np.mean([net(Xt).numpy() for net in self.nets_], axis=0)
        p_krr = np.asarray(self.krr_.predict(X), dtype=np.float32)
        if p_krr.ndim == 1:
            p_krr = p_krr[:, None]
        p = self.blend * p_mlp + (1.0 - self.blend) * p_krr
        return p[:, 0] if self._1d else p

def build_model(params: dict, task: str, seed: int):
    return BlendNoisyMLPKRR(seed=seed, **params)
