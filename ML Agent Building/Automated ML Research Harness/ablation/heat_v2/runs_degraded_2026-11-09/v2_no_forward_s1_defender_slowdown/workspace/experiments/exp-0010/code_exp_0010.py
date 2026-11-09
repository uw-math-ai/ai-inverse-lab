import numpy as np
import torch

def _net(d, o):
    L = []
    for a, b in [(d, 512), (512, 256), (256, 128)]:
        L += [torch.nn.Linear(a, b), torch.nn.LayerNorm(b), torch.nn.GELU()]
    L.append(torch.nn.Linear(128, o))
    return torch.nn.Sequential(*L)

class Ens:
    def fit(self, X, y):
        X = np.asarray(X, np.float32); Y = np.log(np.asarray(y, np.float32))
        self.mu = torch.tensor(X.mean(0)); self.sd = torch.tensor(X.std(0) + 1e-6)
        self.ym = Y.mean(0); self.ys = Y.std(0) + 1e-6
        Xt = torch.tensor(X); Yt = torch.tensor((Y - self.ym) / self.ys)
        n, d = X.shape; bs = 128; self.nets = []
        for i in range(self.n):
            torch.manual_seed(1000 * self.seed + 17 * i + self.off)
            net = _net(d, Y.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs * ((n + bs - 1) // bs))
            net.train()
            for ep in range(self.epochs):
                perm = torch.randperm(n)
                for j in range(0, n, bs):
                    b = perm[j:j + bs]; xb = Xt[b] + self.sig * torch.randn(len(b), d)
                    loss = ((net((xb - self.mu) / self.sd) - Yt[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval(); self.nets.append(net)
        return self
    def predict(self, X):
        Xt = (torch.tensor(np.asarray(X, np.float32)) - self.mu) / self.sd
        out = []
        with torch.no_grad():
            for net in self.nets:
                out.append(np.exp(net(Xt).numpy() * self.ys + self.ym))
        return np.mean(out, 0)

def build_model(params, task, seed):
    m = Ens(); m.n = int(params.get('n', 4)); m.epochs = int(params.get('epochs', 200)); m.sig = float(params.get('sigma', 1e-3)); m.off = int(params.get('seed_offset', 1)); m.seed = int(seed)
    return m
