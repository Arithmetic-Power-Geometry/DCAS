import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, binomtest
import matplotlib.pyplot as plt
os.makedirs('tables',exist_ok=True); os.makedirs('figures',exist_ok=True)

def ff_key(r):
    return (0,float(r.best_f),float(r.first_feasible)) if int(r.success)==1 else (1,float(r.final_ncv),float(r.first_feasible))

def add_ranks(df, blockcols):
    chunks=[]
    for _,g in df.groupby(blockcols,sort=False):
        gg=g.copy(); keys=[ff_key(r) for _,r in gg.iterrows()]
        order=sorted(range(len(keys)), key=lambda i:keys[i]); ranks=np.empty(len(keys),float)
        # average ranks for exact ties
        pos=0
        while pos<len(order):
            j=pos+1
            while j<len(order) and keys[order[j]]==keys[order[pos]]: j+=1
            avg=(pos+1+j)/2.0
            for k in range(pos,j): ranks[order[k]]=avg
            pos=j
        gg['rank']=ranks; chunks.append(gg)
    return pd.concat(chunks,ignore_index=True)

# Cross-host campaign
x=pd.read_csv('data/raw/crosshost.csv'); xr=add_ranks(x,['problem','seed'])
mr=xr.groupby('method').agg(mean_rank=('rank','mean'),success=('success','mean'),median_eff=('first_feasible','median'),median_runtime=('elapsed_sec','median')).reset_index().sort_values('mean_rank')
mr.to_csv('tables/crosshost_overall.csv',index=False)
# paired host deltas: DCAS host vs base host on same problem-seed
pairs=[('DCAS-DE','JADE-NCV'),('DCAS-CMA','CMA-ES'),('DCAS-PSO','PSO')]; rows=[]
for a,b in pairs:
    pv=xr[xr.method.isin([a,b])].pivot_table(index=['problem','seed'],columns='method',values='rank',aggfunc='first').dropna()
    d=pv[a]-pv[b]
    try: _,p=wilcoxon(d,alternative='less')
    except: p=1.0
    rows.append(dict(dcas=a,baseline=b,n_blocks=len(d),mean_delta_rank=float(d.mean()),A12=float((np.sum(d<0)+.5*np.sum(d==0))/len(d)),p=float(p)))
pairdf=pd.DataFrame(rows); pairdf.to_csv('tables/crosshost_pairs.csv',index=False)

# direct feasibility-first paired outcomes (pair-only, independent of other methods)
def direct_pair(a,b,df):
    w=l=t=0
    for _,g in df[df.method.isin([a,b])].groupby(['problem','seed']):
        ra=g[g.method==a].iloc[0]; rb=g[g.method==b].iloc[0]
        ka=ff_key(ra); kb=ff_key(rb)
        if ka<kb: w+=1
        elif kb<ka: l+=1
        else: t+=1
    pv=binomtest(w,w+l,.5,alternative='greater').pvalue if (w+l)>0 else 1.0
    return dict(dcas=a,baseline=b,wins=w,losses=l,ties=t,A12_direct=(w+.5*t)/max(w+l+t,1),sign_p=pv)
direct=pd.DataFrame([direct_pair(a,b,x) for a,b in pairs])
order=direct.sign_p.sort_values().index; vals=direct.loc[order,'sign_p'].to_numpy(); m=len(vals); adj=np.maximum.accumulate(np.minimum(1,vals*np.arange(m,0,-1))); direct.loc[order,'holm_p']=adj
direct.to_csv('tables/crosshost_direct_pairs.csv',index=False)

# per-problem success and rank for host pairs
ph=xr.groupby(['problem','method']).agg(mean_rank=('rank','mean'),success=('success','mean'),median_eff=('first_feasible','median'),median_runtime=('elapsed_sec','median')).reset_index(); ph.to_csv('tables/crosshost_problem.csv',index=False)
# plot mean ranks
plt.figure(figsize=(7.2,4.2)); plt.bar(mr.method,mr.mean_rank); plt.xticks(rotation=35,ha='right'); plt.ylabel('Mean matched rank (lower is better)'); plt.tight_layout(); plt.savefig('figures/crosshost_ranks.png',dpi=180); plt.close()
# pair effect plot
plt.figure(figsize=(6.2,4.2)); plt.bar(pairdf.dcas,pairdf.A12); plt.axhline(.5,ls='--'); plt.ylim(0,1); plt.ylabel('A12 vs host baseline'); plt.tight_layout(); plt.savefig('figures/crosshost_effects.png',dpi=180); plt.close()

# Large scale: paired within problem,seed across four methods
l=pd.read_csv('data/raw/large_scale.csv'); lr=add_ranks(l,['problem','seed']); lm=lr.groupby(['dim','method']).agg(mean_rank=('rank','mean'),success=('success','mean'),median_eff=('first_feasible','median'),median_runtime=('elapsed_sec','median')).reset_index(); lm.to_csv('tables/large_scale.csv',index=False)
# aggregate pair effects by dimension and overall
rows=[]
for dim,g in lr.groupby('dim'):
    for a,b in [('DCAS-DE','JADE-NCV'),('DCAS-PSO','PSO')]:
        pv=g[g.method.isin([a,b])].pivot_table(index=['problem','seed'],columns='method',values='rank',aggfunc='first').dropna(); d=pv[a]-pv[b]
        try:_,p=wilcoxon(d,alternative='less')
        except:p=1.0
        rows.append(dict(dim=dim,dcas=a,baseline=b,n=len(d),A12=float((np.sum(d<0)+.5*np.sum(d==0))/len(d)),mean_delta_rank=float(d.mean()),p=float(p)))
pd.DataFrame(rows).to_csv('tables/large_scale_pairs.csv',index=False)
plt.figure(figsize=(7.2,4.4))
for m,g in lm.groupby('method'): plt.plot(g.dim,g.mean_rank,marker='o',label=m)
plt.xscale('log'); plt.xlabel('Dimension'); plt.ylabel('Mean matched rank'); plt.legend(); plt.tight_layout(); plt.savefig('figures/large_scale.png',dpi=180); plt.close()

# ECDF of normalized feasibility time for cross-host
plt.figure(figsize=(6.5,4.2))
for m in ['DCAS-DE','JADE-NCV','DCAS-CMA','CMA-ES','DCAS-PSO','PSO']:
    z=np.sort(np.minimum(1.0,x[x.method==m].first_feasible.to_numpy()/x[x.method==m].budget.to_numpy())); y=np.arange(1,len(z)+1)/len(z); plt.step(z,y,where='post',label=m)
plt.xlabel('Fraction of budget to first feasibility'); plt.ylabel('ECDF'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig('figures/crosshost_eff_ecdf.png',dpi=180); plt.close()
print('CROSSHOST\n',mr.to_string(index=False)); print('\nPAIRS\n',pairdf.to_string(index=False)); print('\nLARGE\n',lm.to_string(index=False))
