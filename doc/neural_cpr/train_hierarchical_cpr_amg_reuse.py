#!/usr/bin/env python3
"""Train a hierarchical CPR -> AMG -> reuse policy from a full joint sweep."""
import argparse, csv, glob, math, re
from collections import defaultdict
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def normalise(row: dict) -> np.ndarray:
    """Match CprPolicyFeatures::toArray(), then select ACTIVE below."""
    def c01(value):
        return float(np.clip(value, 0.0, 1.0))

    return np.array([
        math.log1p(float(row["dt_days"])) / 7.0,
        c01(float(row["dt_ratio"]) / 10.0),
        (math.log10(max(float(row["nl_residual_norm"]), 1e-14)) + 14.0) / 14.0,
        c01(float(row["nl_residual_reduce"])),
        c01(float(row["nl_iteration"]) / 20.0),
        math.log1p(float(row["nnz_per_row"])) / 5.0,
        c01(float(row["diag_dominance"]) / 4.0),
        c01(float(row["prev_linsolver_iters"]) / 50.0),
        float(np.clip(float(row["prev_solve_failed"]), 0.0, 1.0)),
        c01(float(row["time_elapsed_frac"])),
        c01(float(row["num_cells_log"]) / 6.0),
        c01(float(row["block_size"]) / 6.0),
        c01(float(row.get("well_density", "0.0")) * 200.0),
        float(np.clip((math.log10(max(float(row.get("nl_residual_trend", "1.0")), 0.01)) + 2.0) / 4.0, 0.0, 1.0)),
        c01(float(row.get("num_phases", "3")) / 3.0),
        c01(float(row.get("dt_cut_count", "0")) / 5.0),
        c01(float(row.get("prev2_linsolver_iters", "0")) / 50.0),
        float(np.clip(math.log10(max(float(row.get("condition_number_estimate", "1.0")), 1.0)) / 10.0, 0.0, 1.0)),
        c01(float(row.get("bhp_well_fraction", "0.0"))),
    ], dtype=np.float32)

INTERVALS=[8,16,32,64,128,256]; DAMPINGS=[1.0,1.3,1.6]; TARGETS=[300,600,1200]
NC,NA,NR=12,9,6
# Deliberately restricted feature set for the CPR/AMG/reuse experiment.
ACTIVE=[0,1,2,3,4,6,7,8]

def load(pattern):
    costs=defaultdict(float); feats={}
    for path in glob.glob(pattern):
        m=re.search(r'_label_(\d+)_pd([^_]+)_ct(\d+)_ri(\d+)\.csv$',path)
        if not m: continue
        c=float(m[2]); t=int(m[3]); r=int(m[4]); label=int(m[1])
        a=DAMPINGS.index(c)*3+TARGETS.index(t); ri=INTERVALS.index(r)
        deck = path[:m.start()]
        with open(path,newline='') as fh:
            for row in csv.DictReader(fh):
                e=int(row['episode'])
                key=(deck,e)
                costs[key,label,a,ri]+=float(row['total_ms'])
                feats.setdefault(key,normalise(row))
    episodes=sorted(feats); X=np.asarray([[feats[e][i] for i in ACTIVE] for e in episodes])
    C=np.full((len(episodes),NC,NA,NR),np.nan); ei={e:i for i,e in enumerate(episodes)}
    for (e,c,a,r),v in costs.items(): C[ei[e],c,a,r]=v
    if np.isnan(C).any(): raise ValueError('incomplete 12x9x6 sweep')
    return episodes,X,C

class Model:
    def __init__(self,nf,h=64,seed=42):
        g=np.random.default_rng(seed); s=lambda shape,scale: g.standard_normal(shape)*scale
        self.W=s((h,nf),math.sqrt(2/nf)); self.b=np.zeros(h)
        self.Wc=s((NC,h),math.sqrt(2/h)); self.bc=np.zeros(NC)
        self.Wa=s((NC,NA,h),math.sqrt(2/h)); self.ba=np.zeros((NC,NA))
        self.Wr=s((NC,NA,NR,h),math.sqrt(2/h)); self.br=np.zeros((NC,NA,NR))
        self.ps=[self.W,self.b,self.Wc,self.bc,self.Wa,self.ba,self.Wr,self.br]
        self.ms=[np.zeros_like(p) for p in self.ps]; self.vs=[np.zeros_like(p) for p in self.ps]; self.step=0
    def forward(self,x):
        z=self.W@x+self.b; h=np.tanh(z)
        c=self.Wc@h+self.bc
        a=np.einsum('cah,h->ca',self.Wa,h)+self.ba
        r=np.einsum('carh,h->car',self.Wr,h)+self.br
        return c[:,None,None]+a[:,:,None]+r,(h,c,a,r)
    def update(self,x,target,lr,temp):
        score,(h,c,a,r)=self.forward(x); q=np.exp((score-score.max()).ravel()); q/=q.sum()
        d=(q-target.ravel()).reshape(NC,NA,NR); loss=float(-np.sum(target*np.log(q.reshape(target.shape)+1e-12)))
        dc=d.sum((1,2)); da=d.sum(2); dh=self.Wc.T@dc+np.einsum('cah,ca->h',self.Wa,da)+np.einsum('carh,car->h',self.Wr,d)
        dz=dh*(1-h*h)
        gs=[np.outer(dz,x),dz,np.outer(dc,h),dc,da[:,:,None]*h[None,None,:],da,d[:,:,:,None]*h[None,None,None,:],d]
        self.step+=1
        for p,g,m,v in zip(self.ps,gs,self.ms,self.vs):
            m[:]=.9*m+.1*g; v[:]=.999*v+.001*g*g
            p[:]-=lr*((m/(1-.9**self.step))/(np.sqrt(v/(1-.999**self.step))+1e-8)+1e-3*p)
        return loss

def target(C,temp):
    q=np.exp(-(C/C.min()-1)/max(temp,1e-5)); return q/q.sum()

def beam(model,x):
    score,(h,c,a,r)=model.forward(x)
    cs=np.argsort(c)[-3:]; candidates=[]
    for ci in cs:
        aa=np.argsort(a[ci])[-3:]
        for ai in aa:
            ri=int(np.argmax(r[ci,ai])); candidates.append((ci,ai,ri,float(score[ci,ai,ri])))
    return max(candidates,key=lambda x:x[3]),candidates

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv-glob',required=True); p.add_argument('--out-prefix',required=True)
    p.add_argument('--epochs',type=int,default=1200); p.add_argument('--patience',type=int,default=200)
    p.add_argument('--hidden',type=int,default=32); p.add_argument('--lr',type=float,default=.002); p.add_argument('--temperature',type=float,default=.03)
    args=p.parse_args(); episodes,X,C=load(args.csv_glob)
    # Hold out the last 15% of each deck independently, preserving temporal
    # ordering while keeping every deck represented in validation.
    tr=[]; va=[]
    for deck in sorted({e[0] for e in episodes}):
        indices=[i for i,e in enumerate(episodes) if e[0]==deck]
        vn=max(1,int(.15*len(indices)))
        tr.extend(indices[:-vn]); va.extend(indices[-vn:])
    tr=np.asarray(tr,dtype=int); va=np.asarray(va,dtype=int)
    model=Model(X.shape[1],args.hidden); rng=np.random.default_rng(42); hist=[]; best=None; bv=np.inf; bad=0
    for ep in range(args.epochs):
        lr=1e-5+.5*(args.lr-1e-5)*(1+math.cos(math.pi*ep/args.epochs)); order=rng.permutation(tr)
        tl=np.mean([model.update(X[i],target(C[i],args.temperature),lr,args.temperature) for i in order])
        vl=np.mean([-np.sum(target(C[i],args.temperature)*np.log(np.exp((model.forward(X[i])[0]-model.forward(X[i])[0].max()))/np.exp(model.forward(X[i])[0]-model.forward(X[i])[0].max()).sum()+1e-12)) for i in va])
        hist.append((tl,vl))
        if vl<bv-1e-5: bv=vl; best=[p.copy() for p in model.ps]; bad=0
        else: bad+=1
        if bad>=args.patience: break
    for p,s in zip(model.ps,best): p[:]=s
    selected=[]; shortlisted=[]; defaults=[]; fastest=[]
    default_a=DAMPINGS.index(1.0)*3+TARGETS.index(1200); default_r=INTERVALS.index(32)
    for i in va:
        pick, cand=beam(model,X[i]); selected.append(C[i,pick[0],pick[1],pick[2]])
        shortlisted.append(min(C[i,c[0],c[1],c[2]] for c in cand)); defaults.append(C[i,0,default_a,default_r]); fastest.append(C[i].min())
    print(f'segments={len(X)} train={len(tr)} val={len(va)} epochs={len(hist)}')
    print('mean measured ms: default %.2f selected %.2f fastest %.2f top3-list %.2f'%(np.mean(defaults),np.mean(selected),np.mean(fastest),np.mean(shortlisted)))
    np.savez(args.out_prefix+'_model.npz',W=model.W,b=model.b,Wc=model.Wc,bc=model.bc,Wa=model.Wa,ba=model.ba,Wr=model.Wr,br=model.br)
    if plt is not None:
        H=np.asarray(hist); fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
        ax[0].plot(H[:,0],label='train loss'); ax[0].plot(H[:,1],label='validation loss'); ax[0].set(title='Hierarchical training loss',xlabel='epoch',ylabel='listwise cross-entropy'); ax[0].legend(frameon=False)
        x=np.arange(len(va)); ax[1].plot(x,defaults,'--',label='default CPR+AMG+reuse'); ax[1].plot(x,fastest,':',label='fastest CPR+AMG+reuse'); ax[1].plot(x,selected,'o-',label='configuration selected by model'); ax[1].plot(x,shortlisted,'s-',label='best database result in 9 model candidates'); ax[1].set(title='Measured time per validation segment',xlabel='segment',ylabel='measured time (ms)'); ax[1].legend(frameon=False,fontsize=8)
        names=['default','model selected','fastest','best in 9 candidates']; vals=[np.mean(defaults),np.mean(selected),np.mean(fastest),np.mean(shortlisted)]
        colors=['#777777','#d95f02','#2ca25f','#3182bd']
        ax[2].barh(names[::-1],vals[::-1],color=colors[::-1]); ax[2].set(title='Mean measured time',xlabel='measured time (ms)');
        for y,v in enumerate(vals[::-1]): ax[2].text(v+.01*max(vals),y,f'{v:.1f} ms',va='center')
        fig.savefig(args.out_prefix+'_metrics.png',dpi=180); print('wrote',args.out_prefix+'_metrics.png')
    else:
        print('matplotlib not installed; skipped metrics plot')
if __name__=='__main__': main()
