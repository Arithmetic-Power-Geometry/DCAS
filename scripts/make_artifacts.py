import os, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, friedmanchisquare
import matplotlib.pyplot as plt
os.makedirs('tables',exist_ok=True); os.makedirs('figures',exist_ok=True)
df=pd.read_csv('data/raw/results.csv')
# Feasibility-first scalar rank within problem-seed: success first, then objective if success else ncv, with EFF tie influence.
def rank_block(g):
    score=[]
    for _,r in g.iterrows():
        if r.success: key=(0,r.best_f,r.first_feasible)
        else:key=(1,r.final_ncv,r.first_feasible)
        score.append(key)
    order=sorted(range(len(score)),key=lambda i:score[i]); ranks=np.empty(len(score),float)
    for j,i in enumerate(order):ranks[i]=j+1
    g=g.copy(); g['rank']=ranks; return g
rd=df.groupby(['problem','seed'],group_keys=False).apply(rank_block,include_groups=False).reset_index(drop=False)
# restore identifying cols possibly stripped by pandas include_groups
if 'problem' not in rd.columns:
    rd=df.merge(rd[['rank']],left_index=True,right_index=True)
meanr=rd.groupby('method')['rank'].mean().sort_values().reset_index(name='mean_rank'); meanr.to_csv('tables/mean_ranks.csv',index=False)
# main summary
summary=df.groupby(['problem','method']).agg(success=('success','mean'),median_eff=('first_feasible','median'),median_best=('best_f',lambda s:np.median(s[np.isfinite(s)]) if np.isfinite(s).any() else np.nan),median_cost=('declared_cost','median')).reset_index(); summary.to_csv('tables/problem_results.csv',index=False)
# paired stats vs DCAS
rows=[]; pivot=rd.pivot_table(index=['problem','seed'],columns='method',values='rank',aggfunc='first')
for m in [c for c in pivot.columns if c!='DCAS']:
    d=pivot['DCAS']-pivot[m]
    try: stat,p=wilcoxon(d,alternative='less')
    except ValueError: stat,p=0,1
    a12=float((np.sum(d<0)+0.5*np.sum(d==0))/len(d))
    rows.append({'comparator':m,'mean_delta_rank':d.mean(),'A12':a12,'p':p})
st=pd.DataFrame(rows).sort_values('p');
# Holm
m=len(st); adj=np.maximum.accumulate(np.minimum(1,st['p'].to_numpy()*np.arange(m,0,-1))); st['holm_p']=adj; st.to_csv('tables/statistics.csv',index=False)
# reward ablation
reward_methods=['DCAS','UCB-Success','UCB-Violation','UCB-Objective']; meanr[meanr.method.isin(reward_methods)].to_csv('tables/reward_ablation.csv',index=False)
# applicability relation: per problem DCAS metrics vs JADE advantage
ap=[]
for p,g in rd.groupby('problem'):
    gp=pivot.loc[p]
    advantage=float((gp['JADE-NCV']-gp['DCAS']).mean())
    base=df[(df.problem==p)&(df.method=='DCAS')]
    ap.append({'problem':p,'allocation_sd':base.allocation_sd.mean(),'rcd':base.rcd.mean(),'advantage_over_JADE':advantage})
ap=pd.DataFrame(ap); rho1,p1=spearmanr(ap['allocation_sd'],ap['advantage_over_JADE']); rho2,p2=spearmanr(ap['rcd'],ap['advantage_over_JADE']); ap.to_csv('tables/applicability.csv',index=False); pd.DataFrame([{'metric':'allocation_sd','rho':rho1,'p':p1},{'metric':'RCD','rho':rho2,'p':p2}]).to_csv('tables/applicability_tests.csv',index=False)
# Friedman over matched blocks for major algorithms
cols=['DCAS','UCB-Success','JADE-NCV','DE-NCV','DE-EPS','DE-PEN']; vals=[pivot[c].values for c in cols]; fr_stat,fr_p=friedmanchisquare(*vals); pd.DataFrame([{'friedman_chi2':fr_stat,'p':fr_p,'n_blocks':len(pivot)}]).to_csv('tables/friedman.csv',index=False)
# figures
plt.figure(figsize=(8,4.5)); plt.bar(meanr.method,meanr.mean_rank); plt.xticks(rotation=45,ha='right'); plt.ylabel('Mean matched rank (lower is better)'); plt.tight_layout(); plt.savefig('figures/mean_ranks.png',dpi=180); plt.close()
succ=df.groupby(['problem','method']).success.mean().reset_index(); keep=['DCAS','UCB-Success','JADE-NCV','DE-NCV']; ps=succ[succ.method.isin(keep)].pivot(index='problem',columns='method',values='success'); ps.plot(kind='bar',figsize=(10,4.8)); plt.ylabel('Feasibility success'); plt.tight_layout(); plt.savefig('figures/success_rates.png',dpi=180); plt.close()
plt.figure(figsize=(5.4,4.2)); plt.scatter(ap.allocation_sd,ap.advantage_over_JADE); [plt.text(r.allocation_sd,r.advantage_over_JADE,r.problem,fontsize=7) for _,r in ap.iterrows()]; plt.xlabel('Operator-allocation heterogeneity'); plt.ylabel('DCAS rank advantage over JADE'); plt.tight_layout(); plt.savefig('figures/heterogeneity.png',dpi=180); plt.close()
plt.figure(figsize=(5.4,4.2)); plt.scatter(ap.rcd,ap.advantage_over_JADE); [plt.text(r.rcd,r.advantage_over_JADE,r.problem,fontsize=7) for _,r in ap.iterrows()]; plt.xlabel('Repair-Constraint Disagreement (RCD)'); plt.ylabel('DCAS rank advantage over JADE'); plt.tight_layout(); plt.savefig('figures/rcd.png',dpi=180); plt.close()
print(meanr.to_string(index=False)); print(st.to_string(index=False)); print('applicability',rho1,p1,rho2,p2,'friedman',fr_stat,fr_p)
# cost study if present
if os.path.exists('data/raw/cost_study.csv'):
    c=pd.read_csv('data/raw/cost_study.csv')
    # quality-cost score: feasibility-first rank within kappa/problem/seed, plus cost efficiency report
    cr=[]
    for (k,p,s),g in c.groupby(['kappa','problem','seed']):
        # rank by success, then objective/ncv; separate cost metric
        keys=[]
        for _,r in g.iterrows(): keys.append((0,r.best_f) if r.success else (1,r.final_ncv))
        order=sorted(range(len(keys)),key=lambda i:keys[i]); ranks=np.empty(len(g));
        for j,i in enumerate(order):ranks[i]=j+1
        gg=g.copy(); gg['rank']=ranks; cr.append(gg)
    cr=pd.concat(cr); cost=cr.groupby(['kappa','method']).agg(mean_rank=('rank','mean'),median_declared_cost=('declared_cost','median'),success=('success','mean')).reset_index(); cost.to_csv('tables/cost_heterogeneity.csv',index=False)
    plt.figure(figsize=(6.5,4.2));
    for m,g in cost.groupby('method'): plt.plot(g.kappa,g.mean_rank,marker='o',label=m)
    plt.xscale('log'); plt.xlabel('Cost heterogeneity kappa'); plt.ylabel('Mean rank (lower is better)'); plt.legend(); plt.tight_layout(); plt.savefig('figures/cost_heterogeneity.png',dpi=180); plt.close()
