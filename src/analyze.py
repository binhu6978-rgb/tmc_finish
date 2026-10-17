"""Offline evidence prediction, with round-blocked resampling and validation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def lines(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def ranks(a):
    a=np.asarray(a);order=np.argsort(a,kind='stable');r=np.empty(len(a),float);j=0
    while j<len(a):
        k=j+1
        while k<len(a) and a[order[k]]==a[order[j]]:k+=1
        r[order[j:k]]=(j+k-1)/2;j=k
    return r


def corr(x,y,rank=False):
    if len(x)<3:return None
    x=np.asarray(x,float);y=np.asarray(y,float)
    if rank:x=ranks(x);y=ranks(y)
    x=x-x.mean();y=y-y.mean();den=np.sqrt((x*x).sum()*(y*y).sum())
    return float((x*y).sum()/den) if den>0 else None


def auc(score,utility):
    x=np.asarray(score);y=np.asarray(utility);mask=y!=0;x=x[mask];y=y[mask]>0
    n=int(y.sum());m=len(y)-n
    if not n or not m:return None
    return float((ranks(x)[y].sum()-n*(n-1)/2)/(n*m))


def metrics(rows,field,bootstrap=True):
    a=[r for r in rows if r.get(field) is not None]
    x=[r[field] for r in a];y=[r['utility_full'] for r in a]
    rounds=sorted(set(r['round'] for r in a))
    result=dict(n=len(a),rounds=len(rounds),positive=sum(v>0 for v in y),negative=sum(v<0 for v in y),
                zero=sum(v==0 for v in y),spearman=corr(x,y,True),pearson=corr(x,y),auroc=auc(x,y),
                status='DESCRIPTIVE_ONLY' if len(a)>=20 and len(rounds)>=3 else 'INSUFFICIENT_SAMPLES',
                spearman_round_bootstrap_95ci=None,auroc_round_bootstrap_95ci=None)
    if bootstrap and len(rounds)>=3:
        rng=np.random.RandomState(81273);s=[];u=[]
        by={t:[r for r in a if r['round']==t] for t in rounds}
        for _ in range(1000):
            b=[r for t in rng.choice(rounds,len(rounds),replace=True) for r in by[t]]
            bx=[r[field] for r in b];byu=[r['utility_full'] for r in b]
            c=corr(bx,byu,True);v=auc(bx,byu)
            if c is not None:s.append(c)
            if v is not None:u.append(v)
        if len(s)>=800:result['spearman_round_bootstrap_95ci']=np.percentile(s,[2.5,97.5]).tolist()
        if len(u)>=800:result['auroc_round_bootstrap_95ci']=np.percentile(u,[2.5,97.5]).tolist()
        result['bootstrap_valid_replicates']=dict(spearman=len(s),auroc=len(u),requested=1000)
    return result


def small_solve(matrix,vector):
    """Pivoted elimination for the tiny ridge system; avoids mixed NumPy/Torch BLAS hangs."""
    a=[list(map(float,row))+[float(v)] for row,v in zip(matrix,vector)];n=len(a)
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(a[i][j]));a[j],a[pivot]=a[pivot],a[j]
        scale=a[j][j]
        if abs(scale)<1e-14:raise ArithmeticError('singular diagnostic ridge system')
        a[j]=[v/scale for v in a[j]]
        for i in range(n):
            if i!=j:
                factor=a[i][j];a[i]=[x-factor*y for x,y in zip(a[i],a[j])]
    return np.array([row[-1] for row in a])


def incremental(rows,field):
    a=[r for r in rows if r.get(field) is not None and r.get('loo_update_cosine') is not None]
    rounds=sorted(set(r['round'] for r in a))
    if len(a)<40 or len(rounds)<6 or min(sum(r['utility_full']>0 for r in a),sum(r['utility_full']<0 for r in a))<5:
        return dict(status='INSUFFICIENT_SAMPLES',n=len(a),rounds=len(rounds))
    x=np.array([[np.log1p(r['sample_count']),np.log1p(r['update_norm']),r['loo_update_cosine'],
                 r['width'],r['depth']]+[float(r['profile']==k) for k in range(7)] for r in a])
    y=np.array([r['utility_full'] for r in a]);e=np.array([r[field] for r in a])
    rr=np.array([r['round'] for r in a]);predictions=[]
    for design in [x,np.column_stack([x,e])]:
        pred=np.zeros(len(a))
        for t in rounds:
            tr=rr!=t;te=~tr;mu=design[tr].mean(0);sd=design[tr].std(0);sd[sd<1e-12]=1
            train=np.column_stack([np.ones(tr.sum()),(design[tr]-mu)/sd])
            test=np.column_stack([np.ones(te.sum()),(design[te]-mu)/sd])
            penalty=np.eye(train.shape[1]);penalty[0,0]=0
            gram=(train[:,:,None]*train[:,None,:]).sum(0)+penalty
            rhs=(train*y[tr,None]).sum(0)
            beta=small_solve(gram,rhs);pred[te]=(test*beta[None,:]).sum(1)
        predictions.append(pred)
    def evaluate(pred):
        pos=y>0;neg=y<0
        return dict(mse=float(((pred-y)**2).mean()),auroc=auc(pred,y),
                    balanced_accuracy=float(((pred[pos]>0).mean()+(pred[neg]<0).mean())/2))
    b,s=map(evaluate,predictions)
    return dict(status='ROUND_HELD_OUT_EXPLORATORY',n=len(a),rounds=len(rounds),
                baseline=b,plus_semantic=s,mse_reduction=b['mse']-s['mse'],
                ridge_penalty=1.,no_hyperparameter_search=True)


def label(m,inc,complete):
    if not complete or m['n']<40 or m['rounds']<6:return 'INSUFFICIENT_EVIDENCE'
    ci=m['spearman_round_bootstrap_95ci'];a=m['auroc'];r=m['spearman']
    if ci and ci[0]>0 and a is not None and a>=.6 and inc.get('mse_reduction',0)>0:
        return 'SEMANTIC_SIGNAL_USEFUL'
    if ci and ci[1]<0 and a is not None and a<.5:return 'SEMANTIC_SIGNAL_MISLEADING'
    if r is not None and r>0 and a is not None and a>.5:return 'SEMANTIC_SIGNAL_WEAK'
    return 'INSUFFICIENT_EVIDENCE'


def summary(rows,key):
    v=[r[key] for r in rows if r.get(key) is not None]
    return dict(n=len(v),mean=float(np.mean(v)) if v else None,
                median=float(np.median(v)) if v else None)


def analyze(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);result={};comparison=[];arms=[];geometry=[]
    for name,folder in [('original',Path(args.original)),('fixed_class',Path(args.fixed))]:
        semantic=lines(folder/'semantic_records.jsonl');utility=lines(folder/'utility_records.jsonl')
        u={(r['round'],r['slot']):r for r in utility}
        joined=[dict(r,**{k:v for k,v in u[r['round'],r['slot']].items() if k not in r})
                for r in semantic if (r['round'],r['slot']) in u]
        status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        complete=status['state']=='complete' and status['rounds_completed']==50 and not cfg['local_check']
        # Score BOTH references on the SAME observations and same non-missing subset.
        common=[r for r in joined if r['original_score'] is not None and r['fixed_score'] is not None]
        associations={k:metrics(joined,k) for k in ['semantic_score','original_score','fixed_score',
                      'prototype_concentration','update_norm','sample_count','profile','width','depth','loo_update_cosine','reliability']}
        common_scores={k:metrics(common,k) for k in ['original_score','fixed_score']}
        inc=incremental(joined,'semantic_score');decision=label(associations['semantic_score'],inc,complete)
        groups={}
        for group in ['Small','Medium','Full','Early','Middle','Late','valid_classes_lt25','valid_classes_25to49','valid_classes_ge50']:
            if group in ['Small','Medium','Full']:selected=[r for r in joined if r['group']==group]
            elif group in ['Early','Middle','Late']:
                selected=[r for r in joined if ('Early' if r['round']<=15 else 'Middle' if r['round']<=32 else 'Late')==group]
            else:
                selected=[r for r in joined if ('valid_classes_lt25' if r['valid_classes']<25 else 'valid_classes_25to49' if r['valid_classes']<50 else 'valid_classes_ge50')==group]
            groups[group]=metrics(selected,'semantic_score')
        high=summary(semantic,'semantic_score')['mean']
        finding='HIGH_SIMILARITY_BUT_LOW_UTILITY_PREDICTIVENESS' if high is not None and high>.9 and decision!='SEMANTIC_SIGNAL_USEFUL' else None
        env=json.loads((folder/'environment.json').read_text(encoding='utf-8'))
        run_summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        result[name]=dict(complete50=complete,status=status,classification=decision,associations=associations,
                         shared_observation_comparison=common_scores,incremental_prediction=inc,groups=groups,
                         config=cfg,environment=env,run_summary=run_summary,
                         joint_training_wall_sum=sum(r.get('joint_training_wall_seconds',0) for r in semantic) if all('joint_training_wall_seconds' in r for r in semantic) else None,
                         prototype_measurement_wall_sum=sum(r.get('prototype_measurement_wall_seconds',0) for r in semantic) if all('prototype_measurement_wall_seconds' in r for r in semantic) else None,
                         auxiliary_finding=finding,semantic_occurrences=len(semantic),utility_occurrences=len(joined))
        for k,m in associations.items():comparison.append(dict(arm=name,signal=k,**m))
        for g in ['Small','Medium','Full']:
            selected=[r for r in semantic if r['group']==g]
            geometry.append(dict(arm=name,group=g,n=len(selected),
                **{k:summary(selected,k) for k in ['valid_classes','comparable_classes','anchor_cold_classes',
                   'original_score','fixed_score','other_class_anchor_cosine','other_class_fixed_cosine']}))
        # Saved native prototypes: distinct physical-client pairs only, no features fabricated.
        import torch
        pair_rows=[]
        for snap in sorted(folder.glob('diagnostic_round_*.pt')):
            snap_data=torch.load(snap,map_location='cpu',weights_only=True)
            representatives={}
            for slot,(cid,p,n,v) in enumerate(snap_data['observations']):
                pid=snap_data['dispatch']['profiles'][slot]
                # Profile IDs have monotonically increasing parameter counts in this family.
                if cid not in representatives or pid>representatives[cid][0]:
                    representatives[cid]=(pid,p,n,v)
            reps=sorted(representatives.items())
            for i,(ci,(pi,p,ni,vi)) in enumerate(reps):
                for cj,(pj,q,nj,vj) in reps[i+1:]:
                    shared=vi&vj;ix=torch.where(vi)[0];jx=torch.where(vj)[0]
                    if not len(ix) or not len(jx):continue
                    cross=p[ix]@q[jx].T;off=ix[:,None]!=jx[None,:]
                    pair_rows.append(dict(arm=name,round=snap_data['round'],client_i=ci,client_j=cj,
                        profile_i=pi,profile_j=pj,shared_valid_classes=int(shared.sum()),
                        same_class_cosine=float((p[shared]*q[shared]).sum(1).mean()) if shared.any() else None,
                        different_class_cosine=float(cross[off].mean()) if off.any() else None))
        if pair_rows:
            with (out/(name+'_cross_profile_geometry.csv')).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
        result[name]['cross_client_geometry']=dict(distinct_client_pairs=len(pair_rows),
            same_class=summary(pair_rows,'same_class_cosine'),different_class=summary(pair_rows,'different_class_cosine'),
            interpretation='Descriptive prototype geometry, not channel alignment or update usefulness')
        actual=lines(folder/'rounds.jsonl')
        arms.append((name,folder,actual))
    a,b=arms[0][2],arms[1][2];n=min(len(a),len(b));pc=cc=0
    for x,y in zip(a,b):pc+=sum(p!=q for p,q in zip(x['profiles'],y['profiles']));cc+=sum(p!=q for p,q in zip(x['clients'],y['clients']))
    integrity=dict(compared_rounds=n,profile_dispatch_mismatch=pc,client_dispatch_mismatch=cc,
                   same_completed_length=len(a)==len(b),both_complete50=all(v['complete50'] for v in result.values()))
    # Direct tensors, once, not a hash framework. Only if both files exist.
    import torch
    ia=torch.load(arms[0][1]/'initial_backbones.pt',map_location='cpu',weights_only=True)
    ib=torch.load(arms[1][1]/'initial_backbones.pt',map_location='cpu',weights_only=True)
    integrity['round0_backbones_equal']=all(torch.equal(x[k],y[k]) for x,y in zip(ia,ib) for k in x)
    # Trusted checkpoints produced by this runner include a Python history dictionary.
    fa=torch.load(arms[0][1]/'latest_state.pt',map_location='cpu',weights_only=False)['global_state']
    fb=torch.load(arms[1][1]/'latest_state.pt',map_location='cpu',weights_only=False)['global_state']
    integrity['final_global_equal']=all(torch.equal(v,fb[k]) for k,v in fa.items()) if len(a)==len(b) else None
    if pc or cc:integrity['valid_paired_dispatch']=False
    else:integrity['valid_paired_dispatch']=True
    def save(name,obj):
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    save('predictive_metrics.json',result);save('dispatch_integrity.json',integrity);save('geometry_comparison.json',geometry)
    for filename,rows in [('semantic_utility_comparison.csv',comparison),('geometry_comparison.csv',geometry)]:
        with (out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
    text=['# Semantic utility comparison','',
          'Local validation outputs are preliminary. Formal server runs require both complete50.',
          'No gate/coverage/view augmentation is used. CE(task)+CE(semantic) jointly trains the backbone.',
          'B changes scoring reference only; fixed directions do not supervise or align embeddings.',
          'Utility is leave-one-occurrence-out marginal CE, not intrinsic client quality.','',
          '## Completion and dispatch', '', '```json',json.dumps(integrity,indent=2),'```','',
          '| arm | complete50 | utility n | Spearman | AUROC | classification |',
          '|---|---|---:|---|---|---|']
    for arm,d in result.items():
        m=d['associations']['semantic_score'];text.append(f"| {arm} | {d['complete50']} | {m['n']} | {m['spearman']} | {m['auroc']} | {d['classification']} |")
    text+=['','## Actual protocol and runtime','',
           '| arm | completed | GPU mapping | train samples | validation n | source commits | gate | wall seconds |',
           '|---|---:|---|---:|---:|---|---|---:|']
    for arm,d in result.items():
        env=d['environment'];text.append(f"| {arm} | {d['status']['rounds_completed']} | physical visible={env['cuda_visible_devices']}, logical={env['logical_device']} | 50000 | {next((r['n_validation'] for r in lines(Path(args.original if arm=='original' else args.fixed)/'utility_records.jsonl')), 'NA')} | baseline619c02e / semantic31c8e8c | false | {d['run_summary']['wall_seconds']:.3f} |")
    text+=['','Train recipe: CIFAR100 cached Dirichlet.3/seed1;100clients/10slots;7profiles;ep5/bs50/dropLast;SGD .01*.998^round,momentum.5,wd1e-4;no augmentation.',
           'Aggregation is paper unnormalized delta with gamma1. Reserved subset is never evaluated.','',
           '## Simple signals and original/fixed scores on the same observations','',
           '| arm | signal | n | positive/negative | Spearman | Pearson | AUROC | round-bootstrap Spearman95CI |',
           '|---|---|---:|---|---|---|---|---|']
    for arm,d in result.items():
        for field,m in d['associations'].items():
            text.append(f"| {arm} | {field} | {m['n']} | {m['positive']}/{m['negative']} | {m['spearman']} | {m['pearson']} | {m['auroc']} | {m['spearman_round_bootstrap_95ci']} |")
    text+=['','## Incremental utility prediction','']
    for arm,d in result.items():text += [f'### {arm}','','```json',json.dumps(d['incremental_prediction'],indent=2),'```','']
    text+=['## Capacity, stage and class-coverage groups','',
           '| arm | group | n | diagnostic rounds | Spearman | AUROC | status |',
           '|---|---|---:|---:|---|---|---|']
    for arm,d in result.items():
        for g,m in d['groups'].items():text.append(f"| {arm} | {g} | {m['n']} | {m['rounds']} | {m['spearman']} | {m['auroc']} | {m['status']} |")
    text+=['','## Geometry and class availability','',
           '| arm | group | occurrences | valid classes mean | comparable classes mean | anchor cold classes mean |',
           '|---|---|---:|---:|---:|---:|']
    for g in geometry:text.append(f"| {g['arm']} | {g['group']} | {g['n']} | {g['valid_classes']['mean']} | {g['comparable_classes']['mean']} | {g['anchor_cold_classes']['mean']} |")
    for arm,d in result.items():text += ['',f"{arm}: cross-client same/different-class geometry (descriptive, no semantic-collapse success criterion):",'```json',json.dumps(d['cross_client_geometry'],indent=2),'```']
    text+=['','## Seven profiles (final diagnostic-validation checkpoint)','',
           '| arm | profile | final validation CE | final validation accuracy |',
           '|---|---:|---:|---:|']
    for arm,folder,actual in arms:
        if actual:
            for pid,v in enumerate(actual[-1]['validation']):text.append(f"| {arm} | {pid} | {v['ce']} | {v['accuracy']} |")
    text+=['','## Interpretation boundaries','',
           'Intervals resample whole rounds (1000 replicates); six planned clusters are few and intervals are exploratory.',
           'Incremental prediction holds out entire rounds, uses fixed ridge penalty1 and fits scaling on training rounds only.',
           'Missing anchor scores stay missing. Shared-observation comparison excludes cold-start missing references in BOTH scores.',
           'Anchor-free concentration is descriptive. A zero neutral reference has undefined cosine and is not assigned a fake score.',
           'Capability/stage/class-coverage groups with too few observations are INSUFFICIENT_SAMPLES.',
           'Reserved official-test subset is not evaluated. It is historically exposed, not a pristine independent test.',
           'See predictive_metrics.json for round-cluster CIs, class counts, subgroup results and incremental-vs-simple-baseline evidence.']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(integrity));print({a:r['classification'] for a,r in result.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--fixed',required=True)
    p.add_argument('--output',required=True);analyze(p.parse_args())
