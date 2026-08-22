import os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','src'))
from dcas import all_problems, run_dcas
PROBLEMS={p.name:p for p in all_problems()}

def task(t):
    kappa,pname,method,reward,seed=t
    p=PROBLEMS[pname]
    r=run_dcas(p,seed=1000+seed,budget=400,reward=reward,discounted=True,cost_kappa=kappa)
    return {'kappa':kappa,'problem':pname,'method':method,'seed':seed,'success':int(r.feasible),'first_feasible':r.first_feasible,'best_f':r.best_f,'final_ncv':r.final_ncv,'declared_cost':r.total_declared_cost,'structural_gain':r.total_structural_gain}

if __name__=='__main__':
    problems=['Annulus10','RotatedBox10','NarrowCorridor12']
    tasks=[]
    for kappa in [1,5,10,50]:
      for pname in problems:
        for method,reward in [('DCAS','structural'),('UCB-Success','success'),('UCB-Violation','violation')]:
          for seed in range(3): tasks.append((kappa,pname,method,reward,seed))
    with ProcessPoolExecutor(max_workers=8) as ex: rows=list(ex.map(task,tasks))
    os.makedirs('data/raw',exist_ok=True); pd.DataFrame(rows).to_csv('data/raw/cost_study.csv',index=False); print('wrote',len(rows),'rows')
