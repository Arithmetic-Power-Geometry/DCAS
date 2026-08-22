from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/external/results.csv'
if not p.exists():
    print('No executed external result file found; analysis skipped without inventing values.')
    raise SystemExit(0)
df=pd.read_csv(p)
req=['suite','problem','dimension','seed','method','eval_budget','success','final_violation','best_f','first_feasible','runtime_s','source_commit','evaluator_id']
missing=[c for c in req if c not in df.columns]
if missing: raise SystemExit('Missing columns: '+','.join(missing))
# feasibility-first rank key; feasible objective then EFF, infeasible violation
rank_rows=[]
for keys,g in df.groupby(['suite','problem','dimension','seed'],sort=False):
    gg=g.copy()
    # lexicographic via tuples then dense ordinal rank; lower better
    vals=[]
    for _,r in gg.iterrows():
        vals.append((0,float(r.best_f),float(r.first_feasible)) if int(r.success)==1 else (1,float(r.final_violation),float(r.first_feasible)))
    order=sorted(range(len(vals)), key=lambda i: vals[i])
    ranks=np.empty(len(vals),float)
    for pos,i in enumerate(order,1): ranks[i]=pos
    gg['rank']=ranks; rank_rows.append(gg)
r=pd.concat(rank_rows,ignore_index=True)
tab=r.groupby('method').agg(mean_rank=('rank','mean'),success=('success','mean'),median_eff=('first_feasible','median'),median_time_s=('runtime_s','median'),runs=('rank','size')).sort_values('mean_rank')
out=ROOT/'tables/external_overall.csv'; tab.reset_index().to_csv(out,index=False)
# paired comparisons against DCAS-DE
base='DCAS-DE'; pairs=[]
for m in sorted(set(r.method)-{base}):
    a=r[r.method==base][['suite','problem','dimension','seed','rank']].rename(columns={'rank':'a'})
    b=r[r.method==m][['suite','problem','dimension','seed','rank']].rename(columns={'rank':'b'})
    z=a.merge(b,on=['suite','problem','dimension','seed'])
    if len(z)==0: continue
    d=z.a-z.b; wins=int((d<0).sum()); losses=int((d>0).sum()); ties=int((d==0).sum()); a12=(wins+.5*ties)/len(z)
    try: pv=wilcoxon(d,alternative='less').pvalue if np.any(d!=0) else 1.0
    except Exception: pv=1.0
    pairs.append({'comparator':m,'blocks':len(z),'wins':wins,'losses':losses,'ties':ties,'A12':a12,'p':pv,'mean_delta_rank':d.mean()})
pairs=pd.DataFrame(pairs)
if len(pairs):
    # Holm correction
    idx=np.argsort(pairs.p.values); adj=np.empty(len(pairs)); last=0
    for k,i in enumerate(idx):
        val=min(1.0,(len(pairs)-k)*pairs.p.iloc[i]); last=max(last,val); adj[i]=last
    pairs['holm_p']=adj; pairs.to_csv(ROOT/'tables/external_pairs.csv',index=False)
# ECDF of fraction budget to first feasible
plt.figure(figsize=(6.3,4.2))
for m,g in r.groupby('method'):
    x=np.sort(np.minimum(1,g.first_feasible/g.eval_budget)); y=np.arange(1,len(x)+1)/len(x); plt.step(x,y,where='post',label=m)
plt.xlabel('Fraction of official evaluation budget to first feasibility'); plt.ylabel('ECDF'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(ROOT/'figures/external_eff_ecdf.png',dpi=180); plt.close()
# performance profile on ranks / best rank per block
wide=r.pivot_table(index=['suite','problem','dimension','seed'],columns='method',values='rank')
ratio=wide.div(wide.min(axis=1),axis=0)
plt.figure(figsize=(6.3,4.2))
xs=np.linspace(1,max(2,float(np.nanmax(ratio.values))),100)
for m in ratio.columns:
    ys=[np.nanmean(ratio[m].values<=x) for x in xs]; plt.plot(xs,ys,label=m)
plt.xlabel('Performance ratio tau'); plt.ylabel('Fraction of blocks'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(ROOT/'figures/external_performance_profile.png',dpi=180); plt.close()
print(tab)
