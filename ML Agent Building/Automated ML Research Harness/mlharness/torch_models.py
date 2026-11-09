"""PyTorch models wrapped as sklearn estimators (torch imported lazily)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class TorchMLP(BaseEstimator):
    def __init__(self, task="regression", hidden=(256, 256), activation="gelu", dropout=0.0,
                 lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=128, patience=20,
                 layer_norm=True, device="cpu", seed=0, n_threads=1):
        self.task = task
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.layer_norm = layer_norm
        self.device = device
        self.seed = seed
        self.n_threads = n_threads

    def _net(self, d_in, d_out):
        import torch.nn as nn
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        layers, d = [], d_in
        for h in self.hidden:
            layers.append(nn.Linear(d, int(h)))
            if self.layer_norm:
                layers.append(nn.LayerNorm(int(h)))
            layers.append(acts.get(self.activation, nn.GELU)())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = int(h)
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        torch.set_num_threads(max(1, int(self.n_threads)))
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()
                                           or self.device == "mps") else "cpu")
        X = np.asarray(X, dtype=np.float32)
        if self.task == "classification":
            self.classes_, y_idx = np.unique(np.asarray(y).ravel(), return_inverse=True)
            Y = y_idx.astype(np.int64)
            d_out = len(self.classes_)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            Y = np.asarray(y, dtype=np.float32)
            self._y1d = Y.ndim == 1
            Y = Y.reshape(len(Y), -1)
            d_out = Y.shape[1]
            loss_fn = torch.nn.MSELoss()
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(1, int(0.1 * n)) if n >= 50 else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xt = torch.from_numpy(X).to(dev)
        Yt = torch.from_numpy(Y).to(dev)
        net = self._net(X.shape[1], d_out).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(self.epochs)))
        best, best_state, bad = np.inf, None, 0
        bs = max(8, int(self.batch_size))
        for epoch in range(int(self.epochs)):
            net.train()
            order = torch.from_numpy(rng.permutation(tr_idx)).to(dev)
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
            sched.step()
            if n_val:
                net.eval()
                with torch.no_grad():
                    v = loss_fn(net(Xt[val_idx]), Yt[val_idx]).item()
                if v < best - 1e-7:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.model_ = net.cpu().eval()
        self.epochs_run_ = epoch + 1
        self.n_params_ = int(sum(p.numel() for p in net.parameters()))
        return self

    def _forward(self, X):
        import torch
        with torch.no_grad():
            return self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()

    def predict(self, X):
        out = self._forward(X)
        if self.task == "classification":
            return self.classes_[out.argmax(axis=1)]
        return out.ravel() if getattr(self, "_y1d", False) else out

    def predict_proba(self, X):
        out = self._forward(X)
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
