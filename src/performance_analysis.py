"""Offline peak-first tables and fixed-round counterfactual diagnostics; no training."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from .analyze import lines,corr,auc,incremental
from .intervention import distribution


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def table(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def association(rows,field):
    rows=[r for r in rows if r.get(field) is not None]
    x=[r[field] for r in rows];y=[r['utility_full'] for r in rows]
    rr=sorted(set(r['round'] for r in rows));ranking=[]
    for t in rr:
        a=[r for r in rows if r['round']==t]
        c=corr([r[field] for r in a],[r['utility_full'] for r in a],True)
        if c is not None:ranking.append(c)
    return dict(n=len(rows),diagnostic_rounds=len(rr),spearman=corr(x,y,True),pearson=corr(x,y),
        auroc_positive_utility=auc(x,y),within_round_spearman_mean=statistics.mean(ranking) if ranking else None,
        within_round_defined=len(ranking),status='EXPLORATORY' if len(rr)>=6 else 'INSUFFICIENT_ROUND_CLUSTERS')


def analyze(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results={};peaks=[];gate_summary=[];utilities=[];profile_rows=[];decisions=[];profile_activity=[]
    for arm,folder in [('b0',Path(args.b0)),('b1',Path(args.b1))]:
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        rounds=lines(folder/'rounds.jsonl');gates=lines(folder/'gate_records.jsonl')
        semantic=lines(folder/'semantic_records.jsonl');clients=lines(folder/'client_semantic_records.jsonl')
        utility=lines(folder/'utility_records.jsonl')
        occurrence={(r['round'],r['slot']):r for r in semantic}
        physical={(r['round'],r['client']):r for r in clients}
        joined=[]
        for u in utility:
            key=(u['round'],u.get('slot',u['client']))
            source=(occurrence if u['unit']=='leave_one_occurrence_out' else physical)[key]
            r=dict(source);r.update(u);joined.append(r)
            r['negative_R']=-r['R']
        fields=['original_score','fixed_score','R','negative_R','reliability','sample_count','valid_classes','update_norm','loo_update_cosine']
        for unit in ['leave_one_occurrence_out','leave_physical_client_out']:
            for group in ['All','Small','Medium','Full','Mixed','Early','Middle','Late']:
                a=[r for r in joined if r['unit']==unit]
                if group in ['Early','Middle','Late']:
                    a=[r for r in a if ('Early' if r['round']<100 else 'Middle' if r['round']<200 else 'Late')==group]
                elif group!='All':a=[r for r in a if r['group']==group]
                for field in fields:utilities.append(dict(arm=arm,unit=unit,group=group,signal=field,**association(a,field)))
        for p in summary['profiles']:profile_rows.append(dict(arm=arm,**p))
        peaks.append(dict(variant=arm,rounds=summary['rounds'],formal_complete300=summary['formal_complete300'],**summary['full']))
        slots=[s for r in gates for s in r['slots']];sl=[s for r in gates for s in r['slot_layers']]
        layer_fraction=sum(s['active'] for s in sl)/len(sl) if sl else 0.
        slot_fraction=sum(s['active'] for s in slots)/len(slots) if slots else 0.
        inactive=slot_fraction<.01 or all(r['aggregate_change_ratio']<1e-6 for r in gates)
        attenuated=sum(r['aggregate_norm_ratio'] is not None and
            (r['aggregate_norm_ratio']<.8 or (r['aggregate_norm_ratio']<.95 and (r['aggregate_cosine'] or 0)>.9999)) for r in gates)
        finding='MEASUREMENT_ONLY' if arm=='b0' else 'GATE_INACTIVE' if inactive else 'GLOBAL_ATTENUATION_RISK' if attenuated>len(gates)/2 else 'DIRECTIONAL_GATE_ACTIVE'
        windows={}
        for stage in ['Early','Middle','Late']:
            own=[r for r in gates if ('Early' if r['round']<100 else 'Middle' if r['round']<200 else 'Late')==stage]
            ownsl=[s for r in own for s in r['slot_layers']]
            windows[stage]=dict(rounds=len(own),rho=distribution([s['rho'] for s in ownsl]),
                Q=distribution([s['Q'] for s in ownsl]),D=distribution([s['D'] for s in ownsl]),
                retention=distribution([s['gate_retention'] for s in ownsl]),
                aggregate_norm_ratio=distribution([r['aggregate_norm_ratio'] for r in own]),
                aggregate_change_ratio=distribution([r['aggregate_change_ratio'] for r in own]))
        groups={g:dict(R=distribution([s['R'] for s in slots if s['group']==g]),
            active_fraction=statistics.mean([float(s['active']) for s in slots if s['group']==g]) if any(s['group']==g for s in slots) else None,
            correction_ratio=distribution([s['correction_ratio'] for s in slots if s['group']==g]),
            correction_l2_sum=sum(s['correction_l2'] for s in slots if s['group']==g)) for g in ['Small','Medium','Full']}
        reliability=sum(r['reliability_valid_unique_clients'] for r in gates)/sum(r['unique_clients'] for r in gates) if gates else None
        for pid in range(7):
            own=[s for s in slots if s['profile']==pid]
            profile_activity.append(dict(arm=arm,profile=pid,slots=len(own),active=sum(s['active'] for s in own),
                active_fraction=sum(s['active'] for s in own)/len(own) if own else None,
                correction_ratio_mean=statistics.mean([s['correction_ratio'] for s in own]) if own else None))
        detail=dict(classification=finding,active_slot_fraction=slot_fraction,active_slot_layer_fraction=layer_fraction,
            rho=distribution([r['rho'] for r in sl]),retention=distribution([r['gate_retention'] for r in sl]),
            Q=distribution([r['Q'] for r in sl]),D=distribution([r['D'] for r in sl]),
            negative_cosines=sum(r['cosine'] is not None and r['cosine']<0 for r in sl),
            valid_cosines=sum(r['cosine'] is not None for r in sl),
            R=distribution([r['R'] for r in slots]),aggregate_norm_ratio=distribution([r['aggregate_norm_ratio'] for r in gates]),
            reliability_valid_fraction_unique_client_round=reliability,windows=windows,profile_groups=groups)
        gate_summary.append(dict(variant=arm,classification=finding,active_slot_fraction=slot_fraction,
            active_slot_layer_fraction=layer_fraction,retention_mean=detail['retention']['mean'],
            retention_min=detail['retention']['min'],norm_ratio_mean=detail['aggregate_norm_ratio']['mean'],
            reliability_valid_fraction=reliability))
        for row in lines(folder/'decision_records.jsonl'):
            if row['decision']!='SHUFFLE_ASSIGNMENT':decisions.append(dict(arm=arm,**row))
        occurrence_rows=[r for r in joined if r['unit']=='leave_one_occurrence_out']
        incremental_evidence={field:incremental(occurrence_rows,field) for field in ['original_score','negative_R']}
        results[arm]=dict(summary=summary,mechanism=detail,incremental_prediction=incremental_evidence)
    a=lines(Path(args.b0)/'rounds.jsonl');b=lines(Path(args.b1)/'rounds.jsonl')
    n=min(len(a),len(b));paired=[]
    for x,y in zip(a,b):
        if x['round']!=y['round'] or x['clients']!=y['clients'] or x['profiles']!=y['profiles']:
            raise ValueError('Paired dispatch differs; cannot report paired training deltas')
        for pid in range(7):
            paired.append(dict(round0=x['round'],profile=pid,b0=x['validation'][pid]['accuracy'],
                b1=y['validation'][pid]['accuracy'],delta=y['validation'][pid]['accuracy']-x['validation'][pid]['accuracy']))
    full=[r['delta'] for r in paired if r['profile']==6]
    d=dict(compared_rounds=n,both_formal_complete300=all(v['summary']['formal_complete300'] for v in results.values()),
        delta_full_peak=results['b1']['summary']['full']['peak']-results['b0']['summary']['full']['peak'],
        delta_full_trajectory=statistics.mean(full) if full else None,
        delta_full_last30=results['b1']['summary']['full']['last30']-results['b0']['summary']['full']['last30'],
        win=sum(x>0 for x in full),tie=sum(x==0 for x in full),lose=sum(x<0 for x in full),
        win_round_ratio=sum(x>0 for x in full)/n if n else None)
    results['comparison']=d
    profile_deltas=[]
    for pid in range(7):
        pa=results['b0']['summary']['profiles'][pid];pb=results['b1']['summary']['profiles'][pid]
        profile_deltas.append(dict(profile=pid,**{f'delta_{k}':pb[k]-pa[k] for k in
            ['peak','top5_mean','top10_mean','final','last30','trajectory_mean']}))
    seven_mean=[dict(round0=x['round'],b0=sum(v['accuracy'] for v in x['validation'])/7,
        b1=sum(v['accuracy'] for v in y['validation'])/7) for x,y in zip(a,b)]
    save(out/'results.json',results);table(out/'full_peak_summary.csv',peaks)
    table(out/'seven_profile_summary.csv',profile_rows);table(out/'gate_summary.csv',gate_summary)
    table(out/'seven_profile_deltas.csv',profile_deltas);table(out/'gate_profile_activity.csv',profile_activity)
    table(out/'seven_profile_mean_trajectory.csv',seven_mean)
    table(out/'semantic_utility_associations.csv',utilities);table(out/'paired_trajectory.csv',paired)
    table(out/'decision_utility.csv',decisions)
    text=['# 300-round Full validation peak comparison','',
        'B0/B1 share joint task+semantic CE, measurement-only Flip and unnormalized delta aggregation.',
        'B1 transfers the historical E5 directional BDRC formula; it is not historical G5 or Locked Method.',
        'No checkpoints/resume. Reserved test is evaluated once after completion at the Full-validation-selected state.',
        'Reserved official-test half was historically exposed; this is an exploratory validation/test partition.','',
        '| variant | rounds | Full peak | round0 / round1 | Top5 mean | Top10 mean | final | last30 | trajectory |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in peaks:text.append(f"| {r['variant']} | {r['rounds']} | {r['peak']:.3f} | {r['peak_round0']} / {r['peak_round1']} | {r['top5_mean']:.3f} | {r['top10_mean']:.3f} | {r['final']:.3f} | {r['last30']:.3f} | {r['trajectory_mean']:.3f} |")
    text+=['','Top5/Top10 mean highest trajectory rounds, not top-k classification accuracy.',
        'Partial/local runs are preliminary: Top5/Top10 use only available rounds; they are not Peak@300 results.','',
        '## Paired comparison','','```json',json.dumps(d,indent=2),'```','',
        '## Mechanism activity','','```json',json.dumps(gate_summary,indent=2),'```','',
        'Gate retention is a directional coefficient (1-rho) on each critical layer, not a whole-client scalar gamma.',
        'GATE_INACTIVE: <1% active slots OR every aggregate relative change <1e-6.',
        'GLOBAL_ATTENUATION_RISK: >half rounds norm ratio <.8, or <.95 with cosine >.9999.',
        'These are fixed reporting flags, not training thresholds; no adaptation follows them.','',
        '## Diagnostic interpretation','',
        'Primary evidence-utility associations use b0 only. b1 has a different evolving model and is secondary.',
        'Every diagnostic intervention uses the same raw-update context. Occurrence removal and all-slots physical-client removal both recompute sample denominators.',
        'Positive LOO utility means including the update lowered validation CE; positive decision delta CE means harm.',
        'Original e is high=consistent; R is high=unstable. R utility AUROC is reported in its raw direction, plus negative_R in the predicted beneficial direction.',
        'Shuffling is round/profile-stratified among valid unique clients, neutral missing risk; duplicates share assigned R in the same profile.',
        'Nonsemantic control matches resulting aggregate L2 by scaling the raw aggregate. Scale may exceed 1 after cancellation; it is not claimed to match individual correction doses.',
        'Ordinary correlations and within-round rankings are descriptive, not independent sample p-values or causal evidence.',
        'A difference in peak alone does not establish semantic selection: retain Top-k, trajectories, nonsemantic/shuffled controls, mechanism inactivity and compression findings.',
        'No mechanism efficacy is established by the short local run. Full 300-round evidence must come from completed server runs.']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(d))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--b0',required=True);p.add_argument('--b1',required=True)
    p.add_argument('--output',required=True);analyze(p.parse_args())
