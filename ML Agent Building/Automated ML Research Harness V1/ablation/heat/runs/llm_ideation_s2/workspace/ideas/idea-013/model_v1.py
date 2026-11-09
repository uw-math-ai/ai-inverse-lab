import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

class NoisyEnsembleMLP(BaseEstimator, RegressorMixin):
    def __init__(self, n_models=3, hidden=384, epochs=220, lr=1.5e-3, wd=5e-4, noise=0.05, seed=0):
        self.n_models=n_models; self.hidden=hidden; self.epochs=epochs; self.lr=lr; self.wd=wd; self.noise=noise; self.seed=seed
    def _net(self, d, o):
        h=self.hidden
        return nn.Sequential(nn.Linear(d,h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h,h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h,h//2), nn.GELU(), nn.Linear(h//2,o))
    def fit(self, X, y):
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
        if y.ndim==1: y=y[:,None]
        n=len(X); rng=np.random.RandomState(self.seed); idx=rng.permutation(n); nv=max(1,int(0.1*n))
        vi, ti = idx[:nv], idx[nv:]
        Xt=torch.tensor(X[ti]); yt=torch.tensor(y[ti]); Xv=torch.tensor(X[vi]); yv=torch.tensor(y[vi])
        self.models_=[]
        torch.set_num_threads(1)
        for m in range(self.n_models):
            torch.manual_seed(self.seed*100+m)
            net=self._net(X.shape[1], y.shape[1])
            opt=torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs*int(np.ceil(len(ti)/128)))
            best=1e18; best_state=None; patience=0
            for ep in range(self.epochs):
                net.train(); perm=torch.randperm(len(ti))
                for b in range(0,len(ti),128):
                    bi=perm[b:b+128]; xb=Xt[bi]+self.noise*torch.randn_like(Xt[bi])
                    loss=nn.functional.smooth_l1_loss(net(xb), yt[bi], beta=0.5)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                net.eval()
                with torch.no_grad(): vl=((net(Xv)-yv)**2).mean().item()
                if vl<best: best=vl; best_state={k:v.clone() for k,v in net.state_dict().items()}; patience=0
                else:
                    patience+=1
                    if patience>40: break
            net.load_state_dict(best_state); net.eval(); self.models_.append(net)
        self.squeeze_=(y.shape[1]==1)
        return self
    def predict(self, X):
        X=torch.tensor(np.asarray(X,dtype=np.float32))
        with torch.no_grad(): p=np.mean([m(X).numpy() for m in self.models_],axis=0)
        return p[:,0] if self.squeeze_ else p

def build_model(params, task, seed):
    p=dict(n_models=3, hidden=384, epochs=220, lr=1.5e-3, wd=5e-4, noise=0.05); p.update(params or {})
    return NoisyEnsembleMLP(seed=seed, **p)
