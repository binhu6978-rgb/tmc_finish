"""E5 ALL_REF BDRC transferred to the probe's unchanged joint training protocol.

This is a directional bounded gate, not the Locked residual client gate or G5.
All tensors here are CPU deltas relative to the actually dispatched model.
"""
import math
import statistics
import torch
from .core import norm, slices

EPS=1e-12


def critical_layers(states,count=3):
    candidates=[]
    for order,(name,v) in enumerate(states[-1].items()):
        if v.ndim!=4 or not name.endswith('weight'):continue
        widths=[s[name].shape[0] for s in states if name in s]
        ratio=max(widths)/min(widths)
        if ratio>1:candidates.append((-ratio,-v.numel(),order,name))
    return [x[-1] for x in sorted(candidates)[:count]]


def covered_median(values,shape):
    """Median of actual contributors, including midpoint for even count; no zero pad votes."""
    stack=torch.full((len(values),*shape),float('nan'),dtype=values[0].dtype)
    for i,x in enumerate(values):stack[(i,*slices(x))]=x
    ordered=stack.sort(dim=0).values
    count=(~stack.isnan()).sum(0);lo=((count-1)//2).clamp(min=0);hi=(count//2).clamp(min=0)
    low=ordered.gather(0,lo.unsqueeze(0)).squeeze(0)
    high=ordered.gather(0,hi.unsqueeze(0)).squeeze(0)
    return torch.where(count>0,(low+high)/2,0.),count>0


def client_risks(records):
    """Historical E5 duplicate policy: count-average class S; max actual count/class."""
    buckets={}
    for r in records:
        target=buckets.setdefault(r['client'],{})
        for c,cosine in r.get('class_flip_cosines',{}).items():
            n=r['class_counts'][int(c)]
            target.setdefault(c,[]).append(((1+cosine)/2,n))
    result={}
    for cid,classes in buckets.items():
        scores=[(sum(s*n for s,n in entries)/sum(n for _,n in entries),max(n for _,n in entries))
                for entries in classes.values()]
        s=sum(s*n for s,n in scores)/sum(n for _,n in scores) if scores else None
        result[cid]=dict(S=s,R=0. if s is None else 1-s,risk_available=s is not None,
                         stability_classes=len(classes))
    return result


def distribution(values):
    values=[float(x) for x in values if x is not None]
    if not values:return dict(n=0,mean=None,median=None,min=None,max=None,p90=None)
    x=torch.tensor(values,dtype=torch.float64)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                min=min(values),max=max(values),p90=float(torch.quantile(x,.9)))


@torch.no_grad()
def bounded_gate(deltas,records,layers,enabled=True,risks=None,slot_risks=None):
    risk_map=client_risks(records) if risks is None else risks
    corrected=list(deltas);layer_rows=[];slot_layers=[]
    for name in layers:
        representatives={}
        for i,r in enumerate(records):
            if name not in deltas[i]:continue
            old=representatives.get(r['client'])
            if old is None or deltas[i][name].numel()>deltas[old][name].numel():
                representatives[r['client']]=i
        unique=list(representatives.values())
        if not unique:continue
        shape=tuple(max(deltas[i][name].shape[axis] for i in unique) for axis in range(deltas[unique[0]][name].ndim))
        ref,coverage=covered_median([deltas[i][name] for i in unique],shape)
        regions={};cos={}
        for i,d in enumerate(deltas):
            if name not in d:continue
            sl=tuple(slice(0,min(a,b)) for a,b in zip(d[name].shape,shape));mask=coverage[sl]
            x=d[name][sl][mask].double();g=ref[sl][mask].double();den=float(x.norm()*g.norm())
            cosine=float(((x*g).sum()/den).clamp(-1,1)) if den>EPS else None
            regions[i]=(sl,mask,x,g);cos[i]=cosine
        votes=[cos[i] for i in unique if cos[i] is not None]
        med=statistics.median(votes) if votes else 0.
        mad=statistics.median([abs(x-med) for x in votes]) if votes else 0.
        quality=max(0.,min(1.,(med-mad+1)/2)) if votes else 0.
        layer_rows.append(dict(layer=name,unique_clients=len(unique),selected_clients=list(representatives),
            selected_slots=unique,reference_norm=float(ref.double().norm()),covered_coordinates=int(coverage.sum()),
            cosine_median=med,cosine_mad=mad,Q=quality))
        for i,(sl,mask,x,g) in regions.items():
            cid=records[i]['client'];rr=(slot_risks[i] if slot_risks is not None else risk_map[cid])['R'];c=cos[i]
            disagreement=(1-c)/2 if c is not None else 0.
            rho=rr*quality*disagreement if enabled and c is not None else 0.
            new=x
            if rho>0 and float(g.square().sum())>EPS:
                alpha=(x*g).sum()/(g.square().sum()+EPS)
                positive=alpha.clamp_min(0)*g
                new=((1-rho)*x+rho*positive).to(deltas[i][name].dtype).double()
                if corrected[i] is deltas[i]:corrected[i]=dict(deltas[i])
                tensor=deltas[i][name].clone();segment=tensor[sl].clone()
                segment[mask]=new.to(tensor.dtype);tensor[sl]=segment;corrected[i][name]=tensor
            change=float((new-x).norm());raw=float(x.norm())
            h0=max(0.,-c) if c is not None else 0.
            h=max(0.,-float((new*g).sum()))/(float(x.norm()*g.norm())+EPS) if c is not None else 0.
            slot_layers.append(dict(slot=i,client=cid,profile=records[i]['profile'],layer=name,
                R=rr,Q=quality,D=disagreement,cosine=c,rho=rho,gate_retention=1-rho,
                active=change>0,raw_l2=raw,corrected_l2=float(new.norm()),correction_l2=change,
                correction_ratio=change/(raw+EPS),negative_residual_before=h0,negative_residual_after=h,
                valid_coordinates=x.numel()))
    slots=[]
    for i,r in enumerate(records):
        original=norm(deltas[i]);change=math.sqrt(sum(float((corrected[i][k]-deltas[i][k]).double().square().sum()) for k in deltas[i]))
        own=[x for x in slot_layers if x['slot']==i]
        slots.append(dict(slot=i,client=r['client'],profile=r['profile'],group=r['group'],
            **(slot_risks[i] if slot_risks is not None else risk_map[r['client']]),
            active=change>0,correction_l2=change,raw_update_l2=original,corrected_update_l2=norm(corrected[i]),
            correction_ratio=change/(original+EPS),active_layers=sum(x['active'] for x in own),
            gate_retention_mean=statistics.mean([x['gate_retention'] for x in own]) if own else 1.))
    return corrected,dict(layers=layer_rows,slot_layers=slot_layers,slots=slots,
        gate_retention=distribution([x['gate_retention'] for x in slot_layers]),
        rho=distribution([x['rho'] for x in slot_layers]),
        active_slots=sum(x['active'] for x in slots),active_slot_layers=sum(x['active'] for x in slot_layers))


def aggregate_difference(base,raw,gated):
    dot=a2=b2=change2=0.
    for k in base:
        a=(raw[k]-base[k]).double();b=(gated[k]-base[k]).double()
        dot+=float((a*b).sum());a2+=float(a.square().sum());b2+=float(b.square().sum())
        change2+=float((a-b).square().sum())
    return dict(raw_aggregate_l2=math.sqrt(a2),gated_aggregate_l2=math.sqrt(b2),
        aggregate_cosine=dot/math.sqrt(a2*b2) if a2*b2>EPS else None,
        aggregate_norm_ratio=math.sqrt(b2/a2) if a2>EPS else None,
        aggregate_change_ratio=math.sqrt(change2/a2) if a2>EPS else 0.)
