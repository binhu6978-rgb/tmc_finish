"""Paired 300-round validation-peak experiments. No model checkpoints or resume.

Diagnostic counterfactuals are evaluated read-only in the current raw-update
context; only scalar records are persisted, never a resumable training state.
"""
import argparse
import csv
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import yaml
from .main import ROOT,write,make_model,build_initial,evaluate
from .core import PROFILES,cpu_state,substate,delta,aggregate,dispatch,norm,loo_cosine
from .data import load_data,partition,evaluation_split
from .semantic import Memory,train_and_measure
from .intervention import bounded_gate,critical_layers,client_risks,aggregate_difference


def trajectory_summary(values):
    x=[float(v) for v in values];ordered=sorted(x,reverse=True);best=x.index(max(x))
    avg=lambda a:sum(a)/len(a) if a else None
    return dict(peak=max(x),peak_round0=best,peak_round1=best+1,final=x[-1],
        top5_mean=avg(ordered[:5]),top10_mean=avg(ordered[:10]),last30=avg(x[-30:]),
        r0_99_mean=avg(x[:100]),r100_199_mean=avg(x[100:200]),r200_299_mean=avg(x[200:300]),
        trajectory_mean=avg(x),topk_count=min(10,len(x)))


def diagnostics(base,uploads,counts,records,layers,test,indices,device,batch,round_id,uf,df):
    """All interventions/omissions use THIS SAME raw, unnormalized base context."""
    raw=aggregate(base,uploads,counts)
    raw_eval=evaluate(raw,6,test,indices,device,batch)
    for i,r in enumerate(records):
        others=[d for j,d in enumerate(uploads) if j!=i];nc=[n for j,n in enumerate(counts) if j!=i]
        without=aggregate(base,others,nc)
        ev=evaluate(without,6,test,indices,device,batch)
        r['loo_update_cosine']=loo_cosine(uploads[i],others,nc)
        uf.write(json.dumps(dict(round=round_id,slot=i,client=r['client'],profile=r['profile'],
            group=r['group'],unit='leave_one_occurrence_out',utility_full=ev['ce']-raw_eval['ce'],
            raw_all_ce=raw_eval['ce'],without_ce=ev['ce'],n_validation=len(indices)),allow_nan=False)+'\n')
    for cid in sorted(set(r['client'] for r in records)):
        remaining=[i for i,r in enumerate(records) if r['client']!=cid]
        without=aggregate(base,[uploads[i] for i in remaining],[counts[i] for i in remaining])
        ev=evaluate(without,6,test,indices,device,batch)
        own=[r['slot'] for r in records if r['client']==cid]
        uf.write(json.dumps(dict(round=round_id,client=cid,slots=own,unit='leave_physical_client_out',
            utility_full=ev['ce']-raw_eval['ce'],raw_all_ce=raw_eval['ce'],without_ce=ev['ce'],
            n_validation=len(indices)),allow_nan=False)+'\n')
    risks=client_risks(records)
    corrected,_=bounded_gate(uploads,records,layers,enabled=True,risks=risks)
    real=aggregate(base,corrected,counts)
    # Deterministic physical-client permutation separately in each actual profile/round.
    # A duplicated client in different profiles participates separately, same-profile
    # duplicates share a risk. Missing remains neutral; only valid risks are shuffled.
    shuffled_records=[];assignment=[]
    for pid in range(7):
        own=[r for r in records if r['profile']==pid]
        unique=sorted(set(r['client'] for r in own if risks[r['client']]['risk_available']))
        values=[risks[c]['R'] for c in unique]
        rng=np.random.RandomState(20261017+round_id*31+pid)
        order=rng.permutation(len(unique)).tolist()
        if len(set(values))>1 and all(values[j]==values[k] for j,k in enumerate(order)):
            order=order[1:]+order[:1]
        mapping={c:dict(risks[c],R=values[k],S=1-values[k]) for c,k in zip(unique,order)}
        assignment.extend(dict(profile=pid,client=c,source_client=unique[k]) for c,k in zip(unique,order))
        for r in own:
            # Keep original physical IDs for reference deduplication; override R only.
            shuffled_records.append((r['slot'],mapping.get(r['client'],risks[r['client']])))
    # Reference and Q must remain exactly the real gate's: replace only each slot R.
    # bounded_gate supports slot-risk overrides but keeps original physical IDs.
    shuffled_updates,_=bounded_gate(uploads,records,layers,enabled=True,risks=risks,
                                   slot_risks=dict(shuffled_records))
    shuffled_state=aggregate(base,shuffled_updates,counts)
    raw_norm=norm({k:raw[k]-base[k] for k in base})
    real_norm=norm({k:real[k]-base[k] for k in base})
    scale=real_norm/raw_norm if raw_norm>1e-12 else 1.
    uniform={k:base[k]+scale*(raw[k]-base[k]) for k in base}
    decision=[]
    for name,state in [('NO_GATE',raw),('REAL_SEMANTIC_GATE',real),
                       ('PROFILE_ROUND_SHUFFLED_GATE',shuffled_state),('NORM_MATCHED_NONSEMANTIC',uniform)]:
        ev=raw_eval if name=='NO_GATE' else evaluate(state,6,test,indices,device,batch)
        row=dict(round=round_id,decision=name,validation_ce=ev['ce'],validation_accuracy=ev['accuracy'],
            delta_ce_vs_raw=ev['ce']-raw_eval['ce'],n_validation=len(indices),
            uniform_scale=scale if name=='NORM_MATCHED_NONSEMANTIC' else None,
            **aggregate_difference(base,raw,state))
        decision.append(row);df.write(json.dumps(row,allow_nan=False)+'\n')
    df.write(json.dumps(dict(round=round_id,decision='SHUFFLE_ASSIGNMENT',assignment=assignment),allow_nan=False)+'\n')
    return decision


def run(args):
    cfg=yaml.safe_load((ROOT/'configs/common.yaml').read_text(encoding='utf-8'))
    cfg.update(yaml.safe_load((ROOT/'configs'/f'performance_300_{args.arm}.yaml').read_text(encoding='utf-8')))
    if args.local_check:
        if not 1<=args.rounds<=3:raise ValueError('Short execution: 1-3 rounds only')
        cfg['rounds']=args.rounds
        cfg['diagnostic_rounds']=[1] if args.rounds>=2 else []
    elif args.rounds!=300 or args.validation_limit:
        raise ValueError('Formal runs require 300 rounds and the full fixed validation subset')
    cfg.update(local_check=args.local_check,arm=args.arm,device=args.device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    completed=0;started=time.perf_counter()
    cfg['data_root']=args.data_root or os.environ.get('CIFAR100_ROOT')
    write(out/'config.json',cfg);write(out/'status.json',dict(state='initializing',rounds_completed=0))
    try:
        if not cfg['data_root']:raise ValueError('Set CIFAR100_ROOT or --data-root')
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        device=torch.device(args.device)
        if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        train,test=load_data(cfg['data_root']);users=partition(ROOT);ev=evaluation_split(test.targets)
        if args.validation_limit:ev['validation']=ev['validation'][:args.validation_limit]
        write(out/'evaluation_split.json',ev)
        plan=dispatch(cfg['seed'],300);write(out/'dispatch_plan.json',plan)
        states=build_initial(cfg['seed']);base=states[-1];layers=critical_layers(states,cfg['critical_layer_count'])
        cfg['critical_layers']=layers;write(out/'config.json',cfg)
        write(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),logical_device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            train_augmentation='none',flip='post_training_measurement_only',coverage_scaling=False,
            aggregation='paper_unnormalized_delta',local_objective='task_CE+semantic_CE',
            saved_model_checkpoints=False,resume_supported=False))
        memory=Memory(cfg);bank=dict(adapters={},classifiers={});history=[]
        best_accuracy=-1.;best_state=None;best_round=None;selected_validation=None
        with (out/'semantic_records.jsonl').open('w',encoding='utf-8') as sf, \
             (out/'client_semantic_records.jsonl').open('w',encoding='utf-8') as cf, \
             (out/'utility_records.jsonl').open('w',encoding='utf-8') as uf, \
             (out/'decision_records.jsonl').open('w',encoding='utf-8') as df, \
             (out/'rounds.jsonl').open('w',encoding='utf-8') as rf, \
             (out/'gate_records.jsonl').open('w',encoding='utf-8') as gf, \
             (out/'accuracy.csv').open('w',encoding='utf-8',newline='') as af:
            aw=csv.writer(af);aw.writerow(['round0','round1','profile','validation_ce','validation_accuracy','n'])
            for t in range(cfg['rounds']):
                tick=time.perf_counter();uploads=[];counts=[];observations=[];records=[]
                for slot,(pid,cid) in enumerate(zip(plan[t]['profiles'],plan[t]['clients'])):
                    with torch.random.fork_rng(devices=[]):net=make_model(pid)
                    net.load_state_dict(states[pid]);net.to(device)
                    p,n,v,stats=train_and_measure(net,train,users[cid],cid,t,slot,cfg,bank,device)
                    local=cpu_state(net);d=delta(local,states[pid])
                    uploads.append(d);counts.append(len(users[cid]));observations.append((cid,p,n,v))
                    records.append(dict(round=t,slot=slot,client=cid,profile=pid,width=PROFILES[pid][0],
                        depth=PROFILES[pid][1],group='Small' if pid<3 else 'Medium' if pid<6 else 'Full',
                        client_tier='Small' if cid<40 else 'Medium' if cid<70 else 'Full',
                        sample_count=counts[-1],update_norm=norm(d),**stats,**memory.score(p,n,v)))
                    del local,net,p,n,v,d
                reliability,merged=memory.reliability(t,observations)
                raw=aggregate(base,uploads,counts)
                gated,gate_log=bounded_gate(uploads,records,layers,enabled=cfg['gate_enabled'])
                actual=aggregate(base,gated,counts) if cfg['gate_enabled'] else raw
                if not all(torch.isfinite(v).all() for v in actual.values()):
                    raise FloatingPointError('Nonfinite aggregated model')
                effect=aggregate_difference(base,raw,actual)
                if t>0 and t in cfg['diagnostic_rounds']:
                    diagnostics(base,uploads,counts,records,layers,test,ev['validation'],device,
                                cfg['eval_batch_size'],t,uf,df)
                risks=client_risks(records)
                for cid,p,n,v in merged:
                    own=[r for r in records if r['client']==cid]
                    groups=sorted(set(r['group'] for r in own))
                    cr=dict(round=t,client=cid,slots=[r['slot'] for r in own],
                        profiles=[r['profile'] for r in own],group=groups[0] if len(groups)==1 else 'Mixed',
                        sample_count=len(users[cid]),valid_classes=int(v.sum()),
                        **memory.score(p,n,v),**reliability[cid],**risks[cid])
                    cf.write(json.dumps(cr,allow_nan=False)+'\n')
                for r in records:
                    r.update(reliability[r['client']]);r.update(risks[r['client']])
                    r.update(semantic_score=r['original_score'],round0_excluded_from_utility=t==0)
                    sf.write(json.dumps(r,allow_nan=False)+'\n')
                gate_log.update(round=t,enabled=cfg['gate_enabled'],gamma_bar=1.,protective_update=False,
                    anchor_valid_classes=int(memory.valid.sum()),
                    reliability_valid_unique_clients=sum(r['reliability'] is not None for r in reliability.values()),
                    unique_clients=len(reliability),**effect)
                gf.write(json.dumps(gate_log,allow_nan=False)+'\n')
                memory.update(merged);base=actual;states=[substate(base,s) for s in states]
                acc=[]
                for pid in range(7):
                    val=evaluate(base,pid,test,ev['validation'],device,cfg['eval_batch_size']);acc.append(val)
                    aw.writerow([t,t+1,pid,val['ce'],val['accuracy'],val['n']])
                if acc[6]['accuracy']>best_accuracy:
                    best_accuracy=acc[6]['accuracy'];best_round=t
                    best_state={k:v.clone() for k,v in base.items()};selected_validation=acc
                completed=t+1
                event=dict(round=t,round1=t+1,clients=plan[t]['clients'],profiles=plan[t]['profiles'],
                    validation=acc,wall_seconds=time.perf_counter()-tick,gate_enabled=cfg['gate_enabled'],
                    seven_profile_mean_accuracy=sum(v['accuracy'] for v in acc)/7,
                    optimizer_steps=sum(r['steps'] for r in records),**effect)
                history.append(event);rf.write(json.dumps(event,allow_nan=False)+'\n')
                for f in [sf,cf,uf,df,rf,gf,af]:f.flush()
                write(out/'status.json',dict(state='running',rounds_completed=completed,
                    full_validation_peak=best_accuracy,peak_round0=best_round))
                print(f"{cfg['experiment']} {completed}/{cfg['rounds']} Full val={acc[6]['accuracy']:.2f} peak={best_accuracy:.2f} {event['wall_seconds']:.1f}s",flush=True)
                del uploads,observations,merged,records,gated,gate_log,raw,actual
        reserved=None
        if not args.local_check:
            # Best Full validation state retained in RAM only. Same global model for all seven.
            reserved=[evaluate(best_state,pid,test,ev['reserved'],device,cfg['eval_batch_size']) for pid in range(7)]
            write(out/'selected_validation_test.json',dict(selection='first_Full_validation_accuracy_max',
                selected_round0=best_round,selected_round1=best_round+1,model_saved=False,
                validation=selected_validation,reserved_test=reserved,
                reserved_is_historically_untouched=False))
        profiles=[dict(profile=pid,width=PROFILES[pid][0],depth=PROFILES[pid][1],
            **trajectory_summary([r['validation'][pid]['accuracy'] for r in history])) for pid in range(7)]
        summary=dict(state='complete',rounds=completed,formal_complete300=not args.local_check and completed==300,
            local_validation_only=args.local_check,full=profiles[6],profiles=profiles,
            full_selected_round0=best_round,full_selected_round1=best_round+1,
            selected_validation_all_profiles=selected_validation,reserved_evaluation_accessed=reserved is not None,
            selected_test_all_profiles=reserved,saved_checkpoints=False,resume_supported=False,
            wall_seconds=time.perf_counter()-started,
            max_memory_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            max_memory_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
        write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',rounds_completed=completed))
    except BaseException as exc:
        write(out/'status.json',dict(state='failed',rounds_completed=completed,error=repr(exc)))
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['b0','b1'],required=True)
    p.add_argument('--rounds',type=int,default=300);p.add_argument('--output',required=True)
    p.add_argument('--data-root',default=os.environ.get('CIFAR100_ROOT'));p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=1);p.add_argument('--local-check',action='store_true')
    p.add_argument('--validation-limit',type=int)
    run(p.parse_args())
