"""Server GPU1 inspection and two short, real, sequential memory measurements."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def gpu_info(index):
    cmd=['nvidia-smi','-i',str(index),'--query-gpu=index,name,memory.total,memory.free','--format=csv,noheader,nounits']
    line=subprocess.check_output(cmd,text=True).strip().split(',')
    return dict(physical_gpu=int(line[0]),name=line[1].strip(),total_mib=int(line[2]),free_mib=int(line[3]))


def main(args):
    root=Path(__file__).resolve().parents[1]
    report=dict(requested_physical_gpu=args.gpu)
    output=root/'logs'/('resource_check_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000))
    output.mkdir(parents=True,exist_ok=False)
    def save():
        (output/'resources.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        report['before']=gpu_info(args.gpu)
        report['other_gpu_processes']=subprocess.check_output(
            ['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'],text=True).strip()
        if args.inspect_only:report['state']='INSPECTED_ONLY';save();print(json.dumps(report,indent=2));return
        # Allow one full training process only with a conservative pre-measurement floor.
        if report['before']['free_mib']<4096:
            report['state']='GPU1_PARALLEL_NOT_SAFE';report['reason']='Less than4096MiB free before single-process measurement'
            save();print(json.dumps(report,indent=2));raise SystemExit(2)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        report['measurements']={}
        for arm in ['original','fixed_class']:
            target=output/arm
            command=[sys.executable,'-u','-B','-m','src.main','--arm',arm,'--rounds','2','--local-check',
                     '--validation-limit','500','--diagnostic-rounds','1','--output',str(target)]
            with (output/(arm+'.log')).open('w',encoding='utf-8') as f:
                proc=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode:
                report['state']='GPU1_PARALLEL_NOT_SAFE';report['reason']='Single-process validation failed: '+arm
                report['exit_code']=proc.returncode;save();raise SystemExit(2)
            report['measurements'][arm]=json.loads((target/'summary.json').read_text(encoding='utf-8'))
        peaks=[v['max_memory_reserved']/1024**2 for v in report['measurements'].values()]
        # PyTorch peak excludes the CUDA context/driver and a second process's context.
        required=sum(peaks)*1.30+2*768+1024
        report['estimate']=dict(measured_reserved_mib=peaks,multiplier=1.30,
                                context_allowance_per_process_mib=768,extra_margin_mib=1024,
                                required_free_mib=required,
                                all_inactive_profiles_on_cpu=True,loo_sequential=True,
                                formal_validation_same_batch128=True)
        report['after']=gpu_info(args.gpu)
        safe=report['after']['free_mib']>=required
        report['state']='GPU1_PARALLEL_SAFE_ESTIMATE' if safe else 'GPU1_PARALLEL_NOT_SAFE'
        save();print(json.dumps(report,indent=2))
        if not safe:raise SystemExit(2)
    except FileNotFoundError:
        report['state']='GPU1_PARALLEL_NOT_SAFE';report['reason']='nvidia-smi or dataset missing';save();raise
    except subprocess.CalledProcessError as exc:
        report['state']='GPU1_PARALLEL_NOT_SAFE';report['reason']='Physical GPU not available: '+str(exc)
        save();print(json.dumps(report,indent=2));raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);p.add_argument('--inspect-only',action='store_true')
    main(p.parse_args())
