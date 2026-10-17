"""Fifty-round server protocol; --local-check performs a separate short validation."""
import argparse
import csv
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import yaml
from .models import ResNet18_cifar
from .core import PROFILES,cpu_state,substate,delta,aggregate,dispatch,norm,loo_cosine
from .data import load_data,partition,evaluation_split
from .semantic import Memory,train_and_measure

ROOT=Path(__file__).resolve().parents[1]


def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def make_model(pid):
    w,d=PROFILES[pid]
    return ResNet18_cifar(num_classes=100,track_running_stats=False,slim_idx=d,scale=w)


def build_initial(seed):
    torch.manual_seed(seed)
    # The original main constructs an unused dense Full before get_model_list.
    # Preserve that initialization RNG consumption without retaining its weights.
    unused=make_model(6)
    del unused
    # CPU only: original seven independent initializations, original profile order.
    return [cpu_state(make_model(p)) for p in range(7)]


@torch.inference_mode()
def evaluate(state,pid,dataset,indices,device,batch_size):
    # Construction/loader RNG is explicitly local; evaluation cannot change training.
    with torch.random.fork_rng(devices=[]):
        net=make_model(pid)
    net.load_state_dict(substate(state,net.state_dict()));net.to(device).eval()
    loader=DataLoader(Subset(dataset,indices),batch_size=batch_size,shuffle=False,
                      generator=torch.Generator().manual_seed(8221),num_workers=0)
    loss=0.;correct=0;n=0
    for x,y in loader:
        x=x.to(device);y=y.to(device);logits=net(x)['output']
        loss+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
        correct+=int((logits.argmax(1)==y).sum());n+=len(y)
    del net
    return dict(ce=loss/n,accuracy=100*correct/n,n=n)


def run(args):
    cfg=yaml.safe_load((ROOT/'configs/common.yaml').read_text(encoding='utf-8'))
    cfg.update(yaml.safe_load((ROOT/'configs'/f'{args.arm}_semantic.yaml').read_text(encoding='utf-8')))
    if args.rounds!=50 and not args.local_check:raise ValueError('short budgets require --local-check')
    if args.local_check and not 1<=args.rounds<=3:raise ValueError('local validation is limited to 1-3 rounds')
    cfg['rounds']=args.rounds;cfg['local_check']=args.local_check
    if args.diagnostic_rounds is not None:
        if not args.local_check:raise ValueError('formal diagnostic rounds are fixed')
        cfg['diagnostic_rounds']=[int(x) for x in args.diagnostic_rounds.split(',') if x]
    if not args.local_check and args.validation_limit:raise ValueError('formal validation uses all5000')
    if cfg['gate_enabled'] is not False:raise ValueError('gate prohibited')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data_root=args.data_root or cfg.get('data_root') or os.environ.get('CIFAR100_ROOT')
    cfg['data_root']=str(data_root) if data_root else None;cfg['device']=args.device
    write(out/'config.json',cfg);write(out/'status.json',dict(state='initializing',rounds_completed=0))
    completed=0;start=time.perf_counter()
    try:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        device=torch.device(args.device)
        if device.type=='cuda':
            torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        if not data_root:raise ValueError('Set CIFAR100_ROOT or --data-root')
        train,test=load_data(data_root);users=partition(ROOT)
        ev=evaluation_split(test.targets)
        if args.validation_limit:ev['validation']=ev['validation'][:args.validation_limit]
        write(out/'evaluation_split.json',ev)
        plan=dispatch(cfg['seed'],50);write(out/'dispatch_plan.json',plan)
        write(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),logical_device=str(device),
              gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
              source_baseline_commit='619c02e',semantic_local_source_commit='31c8e8cf73ea02f509dcf442e91d792c4542b570',
              gate_enabled=False,coverage_scaling=False,train_augmentation='none'))
        states=build_initial(cfg['seed']);base=states[-1]
        torch.save(states,out/'initial_backbones.pt')
        construction_peak=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
        gpu_construction_allocated=0
        bank=dict(adapters={},classifiers={});memory=Memory(cfg)
        with (out/'semantic_records.jsonl').open('w',encoding='utf-8') as sf, \
             (out/'utility_records.jsonl').open('w',encoding='utf-8') as uf, \
             (out/'rounds.jsonl').open('w',encoding='utf-8') as rf, \
             (out/'training.log').open('w',encoding='utf-8') as log, \
             (out/'accuracy.csv').open('w',encoding='utf-8',newline='') as af:
            aw=csv.writer(af);aw.writerow(['round0','profile','validation_ce','validation_accuracy','n'])
            for t in range(args.rounds):
                tick=time.perf_counter();uploads=[];counts=[];observations=[];records=[]
                for slot,(pid,cid) in enumerate(zip(plan[t]['profiles'],plan[t]['clients'])):
                    with torch.random.fork_rng(devices=[]):net=make_model(pid)
                    net.load_state_dict(states[pid]);dispatched=states[pid]
                    net.to(device)
                    if device.type=='cuda':
                        gpu_construction_allocated=max(gpu_construction_allocated,torch.cuda.memory_allocated(device))
                    p,n,v,stats=train_and_measure(net,train,users[cid],cid,t,slot,cfg,bank,device)
                    local=cpu_state(net);d=delta(local,dispatched)
                    if not all(torch.isfinite(x).all() for x in d.values()):raise FloatingPointError('nonfinite upload')
                    uploads.append(d);counts.append(len(users[cid]));observations.append((cid,p,n,v))
                    record=dict(round=t,slot=slot,client=cid,profile=pid,width=PROFILES[pid][0],
                        depth=PROFILES[pid][1],group='Small' if pid<3 else 'Medium' if pid<6 else 'Full',
                        sample_count=counts[-1],update_norm=norm(d),**stats,**memory.score(p,n,v))
                    record['semantic_score']=record['original_score'] if args.arm=='original' else record['fixed_score']
                    records.append(record);del local,net,p,n,v,d
                reliability,merged=memory.reliability(t,observations)
                all_state=aggregate(base,uploads,counts)  # gamma=None always: no semantic intervention
                selected=t>0 and t in cfg['diagnostic_rounds']
                if selected:
                    all_eval=evaluate(all_state,6,test,ev['validation'],device,cfg['eval_batch_size'])
                    for i,record in enumerate(records):
                        other=[d for j,d in enumerate(uploads) if i!=j]
                        nc=[n for j,n in enumerate(counts) if i!=j]
                        without=aggregate(base,other,nc)
                        omit_eval=evaluate(without,6,test,ev['validation'],device,cfg['eval_batch_size'])
                        utility=omit_eval['ce']-all_eval['ce']
                        record['loo_update_cosine']=loo_cosine(uploads[i],other,nc)
                        u=dict(round=t,slot=i,client=record['client'],profile=record['profile'],
                               utility_full=utility,all_ce=all_eval['ce'],without_ce=omit_eval['ce'],
                               sign='POSITIVE' if utility>0 else 'NEGATIVE' if utility<0 else 'ZERO',
                               n_validation=len(ev['validation']),unit='leave_one_occurrence_out')
                        uf.write(json.dumps(u,allow_nan=False)+'\n');del without
                    torch.save(dict(round=t,base=base,deltas=uploads,counts=counts,
                                    dispatch=plan[t],observations=observations,
                                    anchor_before=memory.anchor.clone(),anchor_valid=memory.valid.clone()),
                               out/f'diagnostic_round_{t:02d}.pt')
                for r in records:
                    r.update(reliability[r['client']]);r['round0_excluded_from_utility']=t==0
                    sf.write(json.dumps(r,allow_nan=False)+'\n')
                # Evidence uses old M_t; update M_(t+1) only AFTER actual aggregation.
                memory.update(merged)
                base=all_state
                states=[substate(base,s) for s in states]
                acc=[]
                for pid in range(7):
                    val=evaluate(base,pid,test,ev['validation'],device,cfg['eval_batch_size']);acc.append(val)
                    aw.writerow([t,pid,val['ce'],val['accuracy'],val['n']])
                completed=t+1
                runtime=time.perf_counter()-tick
                event=dict(round=t,profiles=plan[t]['profiles'],clients=plan[t]['clients'],
                           gate_enabled=False,gamma_bar=1.,protection=False,validation=acc,
                           wall_seconds=runtime,optimizer_steps=sum(r['steps'] for r in records))
                rf.write(json.dumps(event,allow_nan=False)+'\n')
                message=f"{cfg['experiment']} round {t+1}/{args.rounds} Full val={acc[-1]['accuracy']:.2f}% {runtime:.1f}s"
                print(message,flush=True);log.write(message+'\n');log.flush()
                sf.flush();uf.flush();rf.flush();af.flush()
                write(out/'status.json',dict(state='running',rounds_completed=completed))
                # Boundary saves include private state; can be inspected, not advertised as a resume API.
                if (t+1)%10==0 or t+1==args.rounds:
                    torch.save(dict(round=t,global_state=base,bank=bank,memory=memory.__dict__),out/'latest_state.pt')
                del uploads,observations,merged,records,all_state
                if device.type=='cuda':torch.cuda.empty_cache()
        summary=dict(state='complete',rounds=completed,formal_complete50=not args.local_check and completed==50,
                     local_validation_only=args.local_check,wall_seconds=time.perf_counter()-start,
                     reserved_evaluation_accessed=False,gate_enabled=False,
                     construction_peak_reserved=construction_peak,
                     gpu_model_construction_allocated_peak=gpu_construction_allocated,
                     max_memory_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                     max_memory_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
        write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',rounds_completed=completed))
    except BaseException as exc:
        write(out/'status.json',dict(state='failed',rounds_completed=completed,error=repr(exc)))
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['original','fixed_class'],required=True)
    p.add_argument('--data-root',default=os.environ.get('CIFAR100_ROOT'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--rounds',type=int,default=50);p.add_argument('--output',required=True)
    p.add_argument('--local-check',action='store_true');p.add_argument('--validation-limit',type=int)
    p.add_argument('--diagnostic-rounds')
    return p


if __name__=='__main__':run(parser().parse_args())
