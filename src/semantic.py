"""Verified initial local semantic training; historical/fixed scoring is read-only."""
from collections import defaultdict
import math
import statistics
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset


def modules(in_dim,classes,dim,device,seed,bank,cid):
    with torch.random.fork_rng(devices=[]):
        a=nn.Linear(in_dim,512,bias=False);q=nn.Linear(dim,classes,bias=True)
        h=nn.Linear(512,dim,bias=False)
        g=torch.Generator().manual_seed(seed+9137)
        weights=torch.linalg.qr(torch.randn(512,dim,generator=g),mode='reduced').Q.T
        with torch.no_grad():
            a.weight.zero_();a.weight[:in_dim,:in_dim].copy_(torch.eye(in_dim))
            q.weight.zero_();q.bias.zero_();h.weight.copy_(weights)
    if (cid,in_dim) in bank['adapters']:a.load_state_dict(bank['adapters'][cid,in_dim])
    if cid in bank['classifiers']:q.load_state_dict(bank['classifiers'][cid])
    h.requires_grad_(False)
    return a.to(device),q.to(device),h.to(device)


def train_and_measure(net,dataset,ids,cid,round_id,slot,cfg,bank,device):
    """Initial git SemConsFL: task CE + semantic CE; no detach at representation.

    Private parameters persist per client/dimension and are never globally aggregated.
    Duplicate occurrences update bank sequentially, matching the initial implementation.
    """
    started=time.perf_counter()
    dim=cfg['embedding_dim'];a,q,h=modules(net.fc.in_features,100,dim,device,cfg['seed'],bank,cid)
    opt=torch.optim.SGD(list(net.parameters())+list(a.parameters())+list(q.parameters()),
        lr=cfg['lr']*cfg['lr_decay']**round_id,momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    seed=cfg['seed']*1000003+round_id*1009+slot*97
    loader=DataLoader(Subset(dataset,ids),batch_size=cfg['batch_size'],shuffle=True,drop_last=True,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    net.train();a.train();q.train();h.eval();steps=0;task=sem=0.
    for _ in range(cfg['local_epochs']):
        for x,y in loader:
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            o=net(x);f=F.adaptive_avg_pool2d(o['representation'],1).flatten(1)
            z=F.normalize(h(a(f)),dim=1,eps=1e-12)
            lt=F.cross_entropy(o['output'],y);ls=F.cross_entropy(q(z),y)
            loss=lt+ls
            if not torch.isfinite(loss):raise FloatingPointError('non-finite local loss')
            loss.backward();opt.step();steps+=1;task+=float(lt.detach());sem+=float(ls.detach())
    bank['adapters'][cid,net.fc.in_features]={k:v.detach().cpu().clone() for k,v in a.state_dict().items()}
    bank['classifiers'][cid]={k:v.detach().cpu().clone() for k,v in q.state_dict().items()}
    trained=time.perf_counter()
    # Deduplicate the ORIGINAL training indices; full native backbone eval, no views.
    net.eval();a.eval();sums=torch.zeros(100,dim,device=device);counts=torch.zeros(100,dtype=torch.long,device=device)
    loader=DataLoader(Subset(dataset,list(dict.fromkeys(ids))),batch_size=cfg['batch_size'],shuffle=False,
                      generator=torch.Generator().manual_seed(seed+31),num_workers=0)
    flip_sums=torch.zeros_like(sums) if cfg.get('measure_flip',False) else None
    clean_forwards=flip_forwards=0
    flip_seconds=0.
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device);y=y.to(device)
            f=F.adaptive_avg_pool2d(net(x)['representation'],1).flatten(1)
            z=F.normalize(h(a(f)),dim=1,eps=1e-12)
            sums.index_add_(0,y,z);counts+=torch.bincount(y,minlength=100)
            clean_forwards+=1
            if flip_sums is not None:
                tick=time.perf_counter()
                f_flip=F.adaptive_avg_pool2d(net(x.flip(-1))['representation'],1).flatten(1)
                z_flip=F.normalize(h(a(f_flip)),dim=1,eps=1e-12)
                flip_sums.index_add_(0,y,z_flip);flip_forwards+=1
                flip_seconds+=time.perf_counter()-tick
    means=sums.cpu()/counts.cpu().clamp(min=1)[:,None];counts=counts.cpu()
    valid=(counts>=cfg['n_proto'])&(means.norm(dim=1)>1e-12)
    p=F.normalize(means,dim=1,eps=1e-12)
    stats=dict(task_ce=task/max(steps,1),semantic_ce=sem/max(steps,1),steps=steps,
               class_counts=counts.tolist(),valid_class_mask=valid.tolist(),
               available_classes=int((counts>0).sum()),valid_classes=int(valid.sum()),
               joint_training_wall_seconds=trained-started,
               prototype_measurement_wall_seconds=time.perf_counter()-trained,
               prototype_concentration=float(means.norm(dim=1)[valid].mean()) if valid.any() else None)
    if flip_sums is not None:
        flip_means=flip_sums.cpu()/counts.clamp(min=1)[:,None]
        flip_valid=valid&(flip_means.norm(dim=1)>1e-12)
        fp=F.normalize(flip_means,dim=1,eps=1e-12)
        cosines=(p*fp).sum(1).clamp(-1,1)
        stability=float((((1+cosines[flip_valid])/2)*counts[flip_valid]).sum()/counts[flip_valid].sum()) if flip_valid.any() else None
        stats.update(stability=stability,risk=None if stability is None else 1-stability,
            class_flip_cosines={str(c):float(cosines[c]) for c in torch.where(flip_valid)[0].tolist()},
            clean_measurement_forwards=clean_forwards,flip_measurement_forwards=flip_forwards,
            flip_measurement_wall_seconds=flip_seconds,probe_deduplicated_samples=int(counts.sum()))
    return p,counts,valid,stats


class Memory:
    def __init__(self,cfg):
        self.cfg=cfg;self.anchor=torch.zeros(100,cfg['embedding_dim']);self.valid=torch.zeros(100,dtype=torch.bool)
        self.support=torch.zeros(100,dtype=torch.long);self.history=defaultdict(list)
        # Same 100 unit directions in d=128 for every client/profile; no learned params.
        self.codes=torch.eye(cfg['embedding_dim'])[:100]

    def score(self,p,n,valid):
        mask=valid & self.valid
        cos=(p*self.anchor).sum(1)
        fixed=(p*self.codes).sum(1)
        def weighted(v,m):
            return float((v[m]*n[m]).sum()/n[m].sum()) if m.any() else None
        # Mismatched-class scores use uniform other VALID references, not future labels.
        def geometry(ref,rv):
            ix=torch.where(valid)[0];j=torch.where(rv)[0]
            if not len(ix) or not len(j):return None
            values=p[ix]@ref[j].T;off=ix[:,None]!=j[None,:]
            return float(values[off].mean()) if off.any() else None
        return dict(original_score=weighted(cos,mask),fixed_score=weighted(fixed,valid),
            anchor_version='round_start',anchor_support_before=self.support.tolist(),
            anchor_initialized_mask=self.valid.tolist(),
            comparable_classes=int(mask.sum()),anchor_cold_classes=int((valid & ~self.valid).sum()),
            class_anchor_cosines={str(c):float(cos[c]) for c in torch.where(mask)[0].tolist()},
            class_fixed_cosines={str(c):float(fixed[c]) for c in torch.where(valid)[0].tolist()},
            other_class_anchor_cosine=geometry(self.anchor,self.valid),
            other_class_fixed_cosine=geometry(self.codes,torch.ones(100,dtype=torch.bool)))

    def reliability(self,t,observations):
        # One physical-client observation/round, original count-weighted merge.
        merged={}
        for cid,p,n,v in observations:
            merged.setdefault(cid,[]).append((p,n,v))
        output={};now=[]
        for cid,items in merged.items():
            p=torch.zeros_like(self.anchor);counts=torch.zeros(100,dtype=torch.long)
            for c in range(100):
                entries=[(x[c],int(n[c])) for x,n,v in items if v[c]]
                if entries:
                    p[c]=F.normalize(sum(x*k for x,k in entries),dim=0,eps=1e-12)
                    counts[c]=max(k for _,k in entries)
            valid=(counts>0)&(p.norm(dim=1)>1e-12);e=self.score(p,counts,valid)['original_score']
            hist=[(r,x) for r,x in self.history[cid] if r>=t-self.cfg['reliability_window']+1]
            if e is not None:hist.append((t,e))
            self.history[cid]=hist
            if len(hist)<2:r=None;status='INSUFFICIENT_HISTORY'
            else:
                a=[x for _,x in hist];r=max(0.,min(1.,statistics.mean(a)-math.sqrt(statistics.variance(a)/len(a))))
                status='AVAILABLE'
            output[cid]=dict(reliability=r,reliability_formula_value=0. if r is None else r,
                             history_count=len(hist),history_status=status)
            now.append((cid,p,counts,valid))
        return output,now

    def update(self,merged):
        # Only after evidence scoring and aggregation, each physical client one vote/class.
        for c in range(100):
            entries=[p[c] for _,p,n,v in merged if v[c]]
            if not entries:continue
            median=torch.quantile(torch.stack(entries),.5,dim=0)
            if median.norm()<=1e-12:continue
            target=F.normalize(median,dim=0,eps=1e-12)
            x=self.cfg['anchor_ema']*self.anchor[c]+(1-self.cfg['anchor_ema'])*target if self.valid[c] else target
            if x.norm()>1e-12:
                self.anchor[c]=F.normalize(x,dim=0,eps=1e-12);self.valid[c]=True
                self.support[c]+=len(entries)
