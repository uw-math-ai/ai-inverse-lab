import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

class _Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.ln = nn.LayerNorm(w)
        self.f = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
    def forward(self, x):
        return x + self.f(self.ln(x))

class _Net(nn.Module):
    def __init__(self, din, dout, w, depth):
        super().__init__()
        self.inp = nn.Linear(din, w)
        self.blocks = nn.Sequential(*[_Block(w) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, dout))
    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))

class ResEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, n_models=3, width=256, depth=3, noise_std=0.05, lr=2e-3, weight_decay=1e-4, epochs=120, batch_size=128, seed=0):
        self.n_models = n_models; self.width = width; self.depth = depth
        self.noise_std = noise_std; self.lr = lr; self.weight_decay = weight_decay
        self.epochs = epochs; self.batch_size = batch_size; self.seed = seed
    def fit(self, X, y):
        torch.set_num_threads(1)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = np.asarray(y)
        self._1d = (y.ndim == 1)
        Y = torch.tensor(y.reshape(len(y), -1), dtype=torch.float32)
        n = X.shape[0]
        self.models_ = []
        for m in range(self.n_models):
            torch.manual_seed(self.seed * 100 + m)
            net = _Net(X.shape[1], Y.shape[1], self.width, self.depth)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            steps = self.epochs * int(np.ceil(n / self.batch_size))
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
            net.train()
            for ep in range(self.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    xb = X[idx] + self.noise_std * torch.randn_like(X[idx])
                    loss = nn.functional.mse_loss(net(xb), Y[idx])
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval()
            self.models_.append(net)
        return self
    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            p = torch.stack([m(X) for m in self.models_]).mean(0).numpy()
        return p.ravel() if self._1d else p

def build_model(params, task, seed):
    return ResEnsemble(seed=seed, **params)
