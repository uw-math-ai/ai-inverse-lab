import numpy as np
import torch
import torch.nn as nn

def _mlp(i,o,h=256):
    return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))

def _train(net,a,b,E,lr=2e-3,bs=128,noise=None):
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(E,1))
    n=a.shape[0]
    for ep in range(E):
        perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]
            xb=a[idx]
            if noise is not None:
                xb=xb+torch.randn_like(xb)*noise
            loss=((net(xb)-b[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    return net

class SurrogateInvert:
    def fit(self,X,y):
        p=self.p
        torch.manual_seed(self.seed)
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        sig=float(p.get('sigma',1e-3))
        self.xm=X.mean(0); self.xs=X.std(0)+1e-8
        t=np.log(np.clip(y,1e-6,None)); self.tm=t.mean(0); self.ts=t.std(0)+1e-8
        Xt=torch.tensor((X-self.xm)/self.xs,dtype=torch.float32)
        Tt=torch.tensor((t-self.tm)/self.ts,dtype=torch.float32)
        n=X.shape[0]; perm=np.random.RandomState(self.seed).permutation(n)
        nh=max(int(0.1*n),50); ho=torch.tensor(perm[:nh]); tr=torch.tensor(perm[nh:])
        d=X.shape[1]; E=int(p.get('epochs',100))
        xs_t=torch.tensor(self.xs,dtype=torch.float32)
        self.f=_train(_mlp(4,d),Tt[tr],Xt[tr],E)
        self.g=_train(_mlp(d,4),Xt[tr],Tt[tr],E,noise=sig/xs_t)
        for q in self.f.parameters(): q.requires_grad_(False)
        for q in self.g.parameters(): q.requires_grad_(False)
        with torch.no_grad():
            r=(self.f(Tt[ho])-Xt[ho])*xs_t
            v=torch.clamp(r.pow(2).mean(0),min=sig*sig)
            self.rho=float(v.mean().sqrt())/sig
            self.w=xs_t/v.sqrt()
            e=(self.g(Xt[ho])-Tt[ho]).pow(2).mean(0).sqrt()
            self.prec=1.0/torch.clamp(e,min=1e-3).pow(2)
        self.lo=torch.tensor((np.log(0.5)-self.tm)/self.ts,dtype=torch.float32)
        self.hi=torch.tensor((np.log(2.0)-self.tm)/self.ts,dtype=torch.float32)
        self.use=False; self.gate=(0.0,0.0)
        if int(p.get('steps',200))>0:
            yh=y[perm[:nh]]
            e0=self._err(self._k(self._enc(Xt[ho])),yh)
            e1=self._err(self._k(self._refine(Xt[ho])),yh)
            self.gate=(e0,e1)
            self.use=bool(e1<0.98*e0)
        return self

    def _err(self,K,Y):
        return float(np.linalg.norm(K-Y)/np.linalg.norm(Y))

    def _k(self,z):
        return np.clip(np.exp(z.numpy()*self.ts+self.tm),0.5,2.0)

    def _enc(self,Xt):
        with torch.no_grad():
            return self.g(Xt)

    def _refine(self,Xt):
        z0=torch.max(torch.min(self._enc(Xt),self.hi),self.lo)
        z=z0.clone().requires_grad_(True)
        opt=torch.optim.Adam([z],lr=float(self.p.get('lr_inv',0.01)))
        a=float(self.p.get('anchor',1.0))
        for s in range(int(self.p.get('steps',200))):
            res=(self.f(z)-Xt)*self.w
            loss=0.5*res.pow(2).sum()+0.5*a*((z-z0).pow(2)*self.prec).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                z.copy_(torch.max(torch.min(z,self.hi),self.lo))
        return z.detach()

    def predict(self,X):
        Xt=torch.tensor((np.asarray(X,dtype=float)-self.xm)/self.xs,dtype=torch.float32)
        z=self._refine(Xt) if self.use else self._enc(Xt)
        return self._k(z)

def build_model(params,task,seed):
    m=SurrogateInvert()
    m.p=dict(params)
    m.seed=int(seed)
    return m
