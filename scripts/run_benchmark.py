import argparse, os, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','src'))
from dcas import all_problems, run_dcas, run_de, run_jade
PROBLEMS={p.name:p for p in all_problems()}

def run_method(p,m,seed,budget,kappa=1.0):
    if m=='DCAS': return run_dcas(p,seed=seed,budget=budget,reward='structural',discounted=True,cost_kappa=kappa)
    if m=='UCB-Success': return run_dcas(p,seed=seed,budget=budget,reward='success',discounted=True,cost_kappa=kappa)
    if m=='UCB-Violation': return run_dcas(p,seed=seed,budget=budget,reward='violation',discounted=True,cost_kappa=kappa)
    if m=='UCB-Objective': return run_dcas(p,seed=seed,budget=budget,reward='objective',discounted=True,cost_kappa=kappa)
    if m=='DCAS-noClosure': return run_dcas(p,seed=seed,budget=budget,reward='structural',use_closure=False,discounted=True,cost_kappa=kappa)
    if m=='DCAS-stationary': return run_dcas(p,seed=seed,budget=budget,reward='structural',discounted=False,cost_kappa=kappa)
    if m=='JADE-NCV': return run_jade(p,seed=seed,budget=budget)
    if m=='DE-NCV': return run_de(p,method='ncv',seed=seed,budget=budget)
    if m=='DE-EPS': return run_de(p,method='eps',seed=seed,budget=budget)
    if m=='DE-PEN': return run_de(p,method='pen',seed=seed,budget=budget)
    raise KeyError(m)

def _task(t):
    pname,m,seed,budget,kappa=t
    p=PROBLEMS[pname]
    r=run_method(p,m,seed,budget,kappa)
    return {'problem':p.name,'dim':p.dim,'method':m,'seed':seed,'budget':budget,'kappa':kappa,'success':int(r.feasible),'first_feasible':r.first_feasible,'best_f':r.best_f,'final_ncv':r.final_ncv,'declared_cost':r.total_declared_cost,'structural_gain':r.total_structural_gain,'allocation_sd':r.applicability.get('allocation_sd',0.0),'rcd':r.applicability.get('rcd',0.0),'action_entropy':r.applicability.get('action_entropy',0.0),'action_counts':json.dumps(r.action_counts)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=30); ap.add_argument('--budget',type=int,default=600); ap.add_argument('--out',default='data/raw/results.csv'); ap.add_argument('--kappa',type=float,default=1.0); ap.add_argument('--jobs',type=int,default=1); ap.add_argument('--methods',default=''); ap.add_argument('--seed-start',type=int,default=0)
    args=ap.parse_args()
    methods=['DCAS','UCB-Success','UCB-Violation','UCB-Objective','DCAS-noClosure','DCAS-stationary','JADE-NCV','DE-NCV','DE-EPS','DE-PEN']
    if args.methods: methods=[m.strip() for m in args.methods.split(',') if m.strip()]
    tasks=[(p.name,m,seed,args.budget,args.kappa) for p in PROBLEMS.values() for m in methods for seed in range(args.seed_start,args.seed_start+args.seeds)]
    if args.jobs<=1:
        rows=[_task(t) for t in tasks]
    else:
        rows=[]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs=[ex.submit(_task,t) for t in tasks]
            for fut in as_completed(futs): rows.append(fut.result())
    rows=sorted(rows,key=lambda r:(r['problem'],r['method'],r['seed']))
    os.makedirs(os.path.dirname(args.out),exist_ok=True); pd.DataFrame(rows).to_csv(args.out,index=False); print(f'wrote {len(rows)} rows -> {args.out}')
if __name__=='__main__': main()
