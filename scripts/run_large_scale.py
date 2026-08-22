import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from dcas import scalable_chain, scalable_shell, run_dcas, run_jade, run_pso, run_dcas_pso

def task(t):
    family,d,m,s,budget,pop=t; p=scalable_chain(d) if family=='chain' else scalable_shell(d); st=time.perf_counter()
    if m=='DCAS-DE': r=run_dcas(p,seed=s,budget=budget,pop_size=pop)
    elif m=='JADE-NCV': r=run_jade(p,seed=s,budget=budget,pop_size=pop)
    elif m=='PSO': r=run_pso(p,seed=s,budget=budget,pop_size=pop)
    elif m=='DCAS-PSO': r=run_dcas_pso(p,seed=s,budget=budget,pop_size=pop)
    sec=time.perf_counter()-st
    return dict(problem=p.name,family=family,dim=d,method=m,seed=s,budget=budget,success=int(r.feasible),first_feasible=r.first_feasible,best_f=r.best_f,final_ncv=r.final_ncv,elapsed_sec=sec)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dims',default='100,500,1000'); ap.add_argument('--seeds',type=int,default=10); ap.add_argument('--budget',type=int,default=1200); ap.add_argument('--pop',type=int,default=40); ap.add_argument('--jobs',type=int,default=1); ap.add_argument('--out',default='data/raw/large_scale.csv'); a=ap.parse_args()
    dims=[int(x) for x in a.dims.split(',')]; methods=['DCAS-DE','JADE-NCV','PSO','DCAS-PSO']; tasks=[(fam,d,m,s,a.budget,a.pop) for fam in ['chain','shell'] for d in dims for m in methods for s in range(a.seeds)]
    if a.jobs<=1: rows=[task(t) for t in tasks]
    else:
        rows=[]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            fs=[ex.submit(task,t) for t in tasks]
            for f in as_completed(fs): rows.append(f.result())
    os.makedirs(os.path.dirname(a.out),exist_ok=True); pd.DataFrame(sorted(rows,key=lambda r:(r['problem'],r['method'],r['seed']))).to_csv(a.out,index=False); print('wrote',len(rows))
if __name__=='__main__': main()
