import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from dcas import all_problems, engineering_problems, run_dcas, run_jade, run_cmaes, run_pso, run_dcas_cma, run_dcas_pso

BASE={p.name:p for p in all_problems()}
for p in engineering_problems(): BASE[p.name]=p
DEFAULT=['G06','G08','Annulus10','RotatedBox10','NarrowCorridor12','WeldedBeam','PressureVessel','TensionSpring','CantileverBeam']

def run_one(t):
    pname,m,seed,budget,pop=t; p=BASE[pname]; st=time.perf_counter()
    if m=='DCAS-DE': r=run_dcas(p,seed=seed,budget=budget,pop_size=pop)
    elif m=='JADE-NCV': r=run_jade(p,seed=seed,budget=budget,pop_size=pop)
    elif m=='CMA-ES': r=run_cmaes(p,seed=seed,budget=budget,pop_size=pop)
    elif m=='DCAS-CMA': r=run_dcas_cma(p,seed=seed,budget=budget,pop_size=pop)
    elif m=='PSO': r=run_pso(p,seed=seed,budget=budget,pop_size=pop)
    elif m=='DCAS-PSO': r=run_dcas_pso(p,seed=seed,budget=budget,pop_size=pop)
    else: raise KeyError(m)
    sec=time.perf_counter()-st
    return dict(problem=pname,dim=p.dim,method=m,seed=seed,budget=budget,success=int(r.feasible),first_feasible=r.first_feasible,best_f=r.best_f,final_ncv=r.final_ncv,declared_cost=r.total_declared_cost,elapsed_sec=sec)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=30); ap.add_argument('--budget',type=int,default=400); ap.add_argument('--pop',type=int,default=40); ap.add_argument('--jobs',type=int,default=1); ap.add_argument('--out',default='data/raw/crosshost.csv'); ap.add_argument('--problems',default=', '.join(DEFAULT))
    a=ap.parse_args(); probs=[x.strip() for x in a.problems.split(',') if x.strip()]; methods=['DCAS-DE','JADE-NCV','CMA-ES','DCAS-CMA','PSO','DCAS-PSO']
    tasks=[(p,m,s,a.budget,a.pop) for p in probs for m in methods for s in range(a.seeds)]
    if a.jobs<=1: rows=[run_one(t) for t in tasks]
    else:
        rows=[]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            fs=[ex.submit(run_one,t) for t in tasks]
            for f in as_completed(fs): rows.append(f.result())
    rows=sorted(rows,key=lambda r:(r['problem'],r['method'],r['seed'])); os.makedirs(os.path.dirname(a.out),exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False); print('wrote',len(rows),a.out)
if __name__=='__main__': main()
