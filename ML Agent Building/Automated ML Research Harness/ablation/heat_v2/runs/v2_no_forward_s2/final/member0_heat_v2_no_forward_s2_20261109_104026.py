import numpy as np
import torch

SIG = 0.001
LO = float(np.log(0.5))
HI = float(np.log(2.0))

def _mlp(i, o, h):
    L = torch.nn
    return L.Sequential(L.Linear(i, h), L.SiLU(), L.Linear(h, h), L.SiLU(), L.Linear(h, h), L.SiLU(), L.Linear(h, o))

def _train(net, A, B, epochs, seed, noise=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = A.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 128):
            idx = perm[s:s + 128]
            a = A[idx]
            if noise is not None:
                a = a + noise * torch.randn_like(a)
            opt.zero_grad()
            loss = ((net(a) - B[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sch.step()
    net.eval()
    return net.double()

def _rel(P, Y):
    return float(np.mean(np.linalg.norm(P - Y, axis=1) / np.linalg.norm(Y, axis=1)))

class EmuLM:
    def __init__(self, params):
        self.p = params
        self.use_refine = False
        self.emu_rmse_raw = None
        self.emu_rel_std = None
        self.crlb_rel = None
        self.crlb_param = None
        self.hold_rel = None

    def _F(self, Z):
        with torch.no_grad():
            out = self.emus[0](Z)
            for e in self.emus[1:]:
                out = out + e(Z)
        return out / len(self.emus)

    def _clamp(self, Z):
        return torch.max(torch.min(Z, self.zhi), self.zlo)

    def _init(self, Xs):
        with torch.no_grad():
            Z = self.inv(torch.tensor(Xs, dtype=torch.float64))
        return self._clamp(Z)

    def _jac(self, Z, F0, h=1e-3):
        cols = []
        for j in range(Z.shape[1]):
            Zp = Z.clone()
            Zp[:, j] = Zp[:, j] + h
            cols.append((self._F(Zp) - F0) / h)
        return torch.stack(cols, dim=2)

    def _out(self, Z):
        return np.clip(np.exp(Z.numpy() * self.zs + self.zm), 0.5, 2.0)

    def _refine(self, Z0, Xs, iters):
        pw = float(self.p.get('prior_w', 0.5))
        sw = self.sw.view(1, -1)
        T = torch.tensor(Xs, dtype=torch.float64)
        Z = Z0.clone()
        n, d = Z.shape
        lam = torch.full((n,), 1e-2, dtype=torch.float64)
        eye = torch.eye(d, dtype=torch.float64).unsqueeze(0)
        F0 = self._F(Z)
        c0 = ((sw * (F0 - T)) ** 2).sum(1)
        for it in range(iters):
            J = self._jac(Z, F0) * sw.unsqueeze(2)
            r = sw * (F0 - T)
            Jt = J.transpose(1, 2)
            H = Jt @ J + pw * eye
            g = (Jt @ r.unsqueeze(2)).squeeze(2) + pw * (Z - Z0)
            A = H + lam.view(-1, 1, 1) * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
            dz = torch.linalg.solve(A, -g.unsqueeze(2)).squeeze(2)
            Zn = self._clamp(Z + dz)
            Fn = self._F(Zn)
            cn = ((sw * (Fn - T)) ** 2).sum(1) + pw * ((Zn - Z0) ** 2).sum(1)
            cz = c0 + pw * ((Z - Z0) ** 2).sum(1)
            acc = cn < cz
            Z = torch.where(acc.view(-1, 1), Zn, Z)
            F0 = torch.where(acc.view(-1, 1), Fn, F0)
            c0 = ((sw * (F0 - T)) ** 2).sum(1)
            lam = torch.where(acc, lam / 3.0, lam * 3.0).clamp(1e-6, 1e6)
        return Z

    def _crlb(self, Z):
        F0 = self._F(Z)
        s = torch.tensor(self.xs / SIG, dtype=torch.float64).view(1, -1, 1)
        J = self._jac(Z, F0) * s
        d = Z.shape[1]
        Fi = J.transpose(1, 2) @ J + 1e-9 * torch.eye(d, dtype=torch.float64).unsqueeze(0)
        C = torch.diagonal(torch.linalg.inv(Fi), dim1=1, dim2=2).numpy()
        K = np.exp(Z.numpy() * self.zs + self.zm)
        vk = K ** 2 * (self.zs ** 2) * C
        per = np.sqrt(vk.sum(1)) / np.linalg.norm(K, axis=1)
        return float(np.mean(per)), np.median(np.sqrt(vk) / K, axis=0).tolist()

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.clip(np.asarray(y, dtype=np.float64), 0.5, 2.0)
        self.xm = X.mean(0)
        self.xs = X.std(0) + 1e-9
        Z = np.log(Y)
        self.zm = Z.mean(0)
        self.zs = Z.std(0) + 1e-9
        Xs = (X - self.xm) / self.xs
        Zs = (Z - self.zm) / self.zs
        self.zlo = torch.tensor((LO - self.zm) / self.zs, dtype=torch.float64)
        self.zhi = torch.tensor((HI - self.zm) / self.zs, dtype=torch.float64)
        n = X.shape[0]
        perm = np.random.default_rng(0).permutation(n)
        nh = max(50, int(0.15 * n))
        ho, tr = perm[:nh], perm[nh:]
        ep = int(self.p.get('epochs', 250))
        h = int(self.p.get('hidden', 192))
        At = torch.tensor(Xs[tr], dtype=torch.float32)
        Bt = torch.tensor(Zs[tr], dtype=torch.float32)
        noise = torch.tensor(SIG / self.xs, dtype=torch.float32)
        self.inv = _train(_mlp(X.shape[1], Y.shape[1], h), At, Bt, ep, 1, noise)
        self.emus = [_train(_mlp(Y.shape[1], X.shape[1], h), Bt, At, ep, 10 + i) for i in range(int(self.p.get('n_emu', 2)))]
        Zh = torch.tensor(Zs[ho], dtype=torch.float64)
        rr = (self._F(Zh).numpy() - Xs[ho]) * self.xs
        ev = np.maximum((rr ** 2).mean(0) - SIG ** 2, 1e-14)
        self.emu_rmse_raw = float(np.sqrt(np.mean(ev)))
        self.emu_rel_std = float(np.sqrt(np.mean(ev / self.xs ** 2)))
        self.sw = torch.tensor(self.xs / np.sqrt(ev + SIG ** 2), dtype=torch.float64)
        try:
            self.crlb_rel, self.crlb_param = self._crlb(Zh)
        except Exception:
            self.crlb_rel, self.crlb_param = None, None
        Zi = self._init(Xs[ho])
        r0 = _rel(self._out(Zi), Y[ho])
        it = int(self.p.get('refine_iters', 20))
        r1 = r0
        if it > 0:
            r1 = _rel(self._out(self._refine(Zi, Xs[ho], it)), Y[ho])
        self.hold_rel = (r0, r1)
        self.use_refine = bool(r1 < 0.97 * r0)
        return self

    def predict(self, X):
        Xs = (np.asarray(X, dtype=np.float64) - self.xm) / self.xs
        Z = self._init(Xs)
        if self.use_refine:
            Z = self._refine(Z, Xs, int(self.p.get('refine_iters', 20)))
        return self._out(Z)

def build_method(params, context):
    return EmuLM(params)
