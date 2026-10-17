"""Paper delta aggregation and original available dispatch; no gate decisions."""
import copy
import random
import numpy as np
import torch

PROFILES = [(0.4,2),(0.4,3),(0.4,4),(0.66,2),(0.66,3),(0.66,4),(1.0,2)]


def slices(x):
    return tuple(slice(0, n) for n in x.shape)


def cpu_state(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def substate(full, template):
    return {k:full[k][slices(v)].clone() for k,v in template.items()}


def delta(local, dispatched):
    return {k:local[k]-dispatched[k] for k in local}


def aggregate(base, deltas, counts, gamma=None, eta=1., floor=.05):
    """Paper Eqs26/27. Recompute total n on EACH leave-one-occurrence-out call.

    gamma is supplied only by mathematical checks. Runtime always uses None=ones.
    No coordinate coverage denominator. Uncovered coordinates have exact zero update.
    """
    if not deltas:
        return {k:v.clone() for k,v in base.items()}
    if len(deltas)!=len(counts) or any(n<=0 for n in counts):
        raise ValueError('invalid occurrence sample counts')
    gamma = [1.] * len(deltas) if gamma is None else gamma
    if len(gamma)!=len(deltas) or any(g<0 or g>1 for g in gamma):
        raise ValueError('invalid gate for numerical check')
    w = [n/sum(counts) for n in counts]
    mass = sum(a*g for a,g in zip(w,gamma))
    scale = 0. if mass==0 else max(1.,floor/mass)
    out = {}
    for k,b in base.items():
        u = torch.zeros_like(b)
        for a,g,d in zip(w,gamma,deltas):
            u[slices(d[k])] += d[k] * (a*g)
        out[k] = b + eta*scale*u
    return out


def loo_cosine(own, others, counts):
    """Training-only external direction; omit current occurrence, use actual overlap."""
    dot=xx=yy=0.
    for k,x in own.items():
        ref=torch.zeros_like(x); mass=torch.zeros_like(x)
        for d,n in zip(others,counts):
            shape=tuple(min(a,b) for a,b in zip(x.shape,d[k].shape))
            sl=tuple(slice(0,a) for a in shape)
            ref[sl]+=d[k][sl]*n;mass[sl]+=n
        valid=mass>0
        a=x[valid].double();b=ref[valid].double()/sum(counts)
        dot+=float((a*b).sum());xx+=float((a*a).sum());yy+=float((b*b).sum())
    return dot/(xx*yy)**.5 if xx>0 and yy>0 else None


def dispatch(seed, rounds):
    # Same random.choice/randint eligibility, private RNGs: explicitly not legacy
    # interleaving with data loaders or semantic-module RNG draws.
    nr=np.random.RandomState(seed); pr=random.Random(seed); result=[]
    for t in range(rounds):
        profiles=nr.choice(range(7),10).tolist()
        clients=[pr.randint(0 if p<3 else 40 if p<6 else 70,99) for p in profiles]
        result.append(dict(round=t,profiles=profiles,clients=clients))
    return result


def norm(state):
    return sum(float((v.double()**2).sum()) for v in state.values())**.5
