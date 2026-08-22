"""Cross-host validation implementations for DCAS.

These hosts keep the DCAS structural-credit/controller abstraction fixed while changing
candidate-generation dynamics. They are intentionally dependency-free so the paper can
be reproduced on GitHub Actions.
"""
from __future__ import annotations
import numpy as np
from .core import (Result, clip, scalar_key, archive_distance, nearest_archive,
                   state_features_cached, state_description_cached, action_cost,
                   constraint_disagreement, DiscountedController, _final, action_entropy)


def _init(problem, seed, pop_size):
    rng=np.random.default_rng(seed); b=problem.bounds; d=problem.dim
    P=rng.uniform(b[:,0],b[:,1],size=(pop_size,d)); vals=[problem.evaluate(x) for x in P]
    farr=np.asarray([z[0] for z in vals],float)
    feas=np.asarray([problem.feasible(z[1]) for z in vals],bool)
    ncv=np.asarray([problem.ncv(z[1]) for z in vals],float)
    archive=[P[i].copy() for i in range(pop_size) if feas[i]][-80:]
    first=next((i+1 for i,z in enumerate(feas) if z),0)
    return rng,b,P,vals,farr,feas,ncv,archive,first


def run_pso(problem, seed=0, pop_size=40, budget=600, w=0.72, c1=1.49, c2=1.49):
    rng,b,P,vals,farr,feas,ncv,archive,first=_init(problem,seed,pop_size)
    d=problem.dim; span=np.maximum(b[:,1]-b[:,0],1e-12); evals=pop_size
    V=rng.normal(0,0.08*span,size=P.shape)
    pbest=P.copy(); pvals=list(vals)
    def best_idx(Avals): return min(range(pop_size), key=lambda j: scalar_key(problem,*Avals[j],'ncv'))
    while evals<budget:
        gi=best_idx(pvals); gb=pbest[gi].copy()
        for i in range(pop_size):
            if evals>=budget: break
            r1=rng.random(d); r2=rng.random(d)
            V[i]=w*V[i]+c1*r1*(pbest[i]-P[i])+c2*r2*(gb-P[i])
            V[i]=np.clip(V[i],-0.25*span,0.25*span)
            trial=clip(P[i]+V[i],b); ft,gt=problem.evaluate(trial); evals+=1
            if problem.feasible(gt) and first==0:first=evals
            if scalar_key(problem,ft,gt,'ncv') <= scalar_key(problem,*pvals[i],'ncv'):
                pbest[i]=trial.copy(); pvals[i]=(ft,gt)
            P[i]=trial; vals[i]=(ft,gt)
    return _final(problem,pbest,pvals,evals,first,{},float(evals),0.0,{'host':'PSO'})


def run_dcas_pso(problem, seed=0, pop_size=40, budget=600, gamma=0.97, beta=0.55, cost_kappa=1.0, reward='structural'):
    rng,b,P,vals,farr,feas,ncv,archive,first=_init(problem,seed,pop_size)
    d=problem.dim; span=np.maximum(b[:,1]-b[:,0],1e-12); evals=pop_size
    V=rng.normal(0,0.06*span,size=P.shape); pbest=P.copy(); pvals=list(vals)
    actions=['balanced','explore','exploit','closure','consensus']; counts={a:0 for a in actions}
    ctrl=DiscountedController(actions,gamma,beta); total_cost=total_gain=0.; recent=[]; asd=[]; rcd=[]
    while evals<budget:
        # feasibility-first best based on personal bests
        order=sorted(range(pop_size),key=lambda j:scalar_key(problem,*pvals[j],'ncv')); gb=pbest[order[0]].copy()
        features=state_features_cached(P,farr,feas,ncv,archive,b,evals,budget,counts,recent)
        elite=P[order[:max(3,pop_size//4)]].mean(axis=0)
        for i in range(pop_size):
            if evals>=budget:break
            fi,gi=vals[i]; available=[a for a in actions if not (a=='closure' and not archive)]
            a=ctrl.choose(available); counts[a]+=1; r1=rng.random(d); r2=rng.random(d)
            if a=='balanced': nv=.72*V[i]+1.49*r1*(pbest[i]-P[i])+1.49*r2*(gb-P[i])
            elif a=='explore': nv=.85*V[i]+.5*r1*(pbest[i]-P[i])+1.1*r2*(gb-P[i])+rng.normal(0,.08*span,d)
            elif a=='exploit': nv=.45*V[i]+.8*r1*(pbest[i]-P[i])+1.8*r2*(gb-P[i])
            elif a=='closure':
                anc=nearest_archive(P[i],archive,b); nv=.35*V[i]+1.7*r2*(anc-P[i])+.4*r1*(pbest[i]-P[i])
            else: nv=.55*V[i]+1.4*r2*(elite-P[i])+.5*r1*(pbest[i]-P[i])
            nv=np.clip(nv,-.3*span,.3*span); trial=clip(P[i]+nv,b); ft,gt=problem.evaluate(trial); evals+=1
            tf=problem.feasible(gt); tn=problem.ncv(gt)
            if tf:
                archive.append(trial.copy()); archive=archive[-80:]
                if first==0:first=evals
            before=state_description_cached(problem,P[i],fi,gi,P,farr,feas,archive,b,features)
            f2a=farr.copy(); f2a[i]=ft; fm=feas.copy(); fm[i]=tf; nn=ncv.copy(); nn[i]=tn
            f2=state_features_cached(P,f2a,fm,nn,archive,b,evals,budget,counts,recent)
            after=state_description_cached(problem,trial,ft,gt,P,f2a,fm,archive,b,f2)
            gain=max(0.,before-after); move=float(np.linalg.norm((trial-P[i])/span))
            base={'balanced':1.,'explore':1.,'exploit':1.,'closure':cost_kappa,'consensus':1.25}[a]
            cost=float(base+.15*move)
            old=scalar_key(problem,fi,gi,'ncv'); new=scalar_key(problem,ft,gt,'ncv')
            if reward=='structural': rr=gain/max(cost,1e-12)
            elif reward=='success': rr=1. if new<old else 0.
            else: rr=max(0.,ncv[i]-tn)/max(cost,1e-12)
            ctrl.update(a,rr); total_cost+=cost; total_gain+=gain
            recent.append((ncv[i]-tn,(fi-ft)/(1+abs(fi)))); recent=recent[-20:]
            P[i]=trial; vals[i]=(ft,gt); farr[i]=ft; feas[i]=tf; ncv[i]=tn; V[i]=nv
            if scalar_key(problem,ft,gt,'ncv') <= scalar_key(problem,*pvals[i],'ncv'):
                pbest[i]=trial.copy(); pvals[i]=(ft,gt)
        alloc=np.array(list(counts.values()),float); alloc/=max(alloc.sum(),1); asd.append(float(np.std(alloc)))
        rcd.append(constraint_disagreement(problem,P,vals,archive,b))
    app={'host':'PSO','allocation_sd':float(np.mean(asd[-5:])) if asd else 0.,'rcd':float(np.mean(rcd[-5:])) if rcd else 0.,'action_entropy':action_entropy(counts)}
    # return best personal-best population for fairness
    return _final(problem,pbest,pvals,evals,first,counts,total_cost,total_gain,app)


def _cma_params(n, lam):
    mu=max(2,lam//2); w=np.log(mu+.5)-np.log(np.arange(1,mu+1)); w/=w.sum(); mueff=1./np.sum(w*w)
    cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
    c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
    damps=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
    return mu,w,mueff,cc,cs,c1,cmu,damps


def run_cmaes(problem, seed=0, pop_size=40, budget=600):
    """Dependency-free full-covariance CMA-ES with feasibility-first ranking."""
    rng=np.random.default_rng(seed); b=problem.bounds; n=problem.dim; span=np.maximum(b[:,1]-b[:,0],1e-12)
    lam=max(6,pop_size); mu,w,mueff,cc,cs,c1,cmu,damps=_cma_params(n,lam)
    m=rng.uniform(b[:,0],b[:,1]); sigma=.22*float(np.mean(span)); C=np.eye(n); pc=np.zeros(n); ps=np.zeros(n); evals=0; first=0
    bestP=[]; bestV=[]; chi=np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    while evals<budget:
        try:
            D2,B=np.linalg.eigh((C+C.T)/2); D=np.sqrt(np.maximum(D2,1e-14)); BD=B*D
        except np.linalg.LinAlgError:
            C=np.eye(n); D=np.ones(n); B=np.eye(n); BD=B
        ys=[]; xs=[]; vals=[]
        for k in range(lam):
            if evals>=budget:break
            z=rng.normal(size=n); y=BD@z; x=clip(m+sigma*y,b); fv,gv=problem.evaluate(x); evals+=1
            if problem.feasible(gv) and first==0:first=evals
            xs.append(x); ys.append((x-m)/max(sigma,1e-12)); vals.append((fv,gv))
            if not bestP or scalar_key(problem,fv,gv,'ncv')<scalar_key(problem,*bestV[0],'ncv'):
                bestP=[x.copy()]; bestV=[(fv,gv)]
        order=sorted(range(len(xs)),key=lambda j:scalar_key(problem,*vals[j],'ncv')); sel=order[:min(mu,len(order))]
        if not sel: break
        ww=w[:len(sel)].copy(); ww/=ww.sum(); y_w=sum(ww[j]*ys[idx] for j,idx in enumerate(sel)); m_old=m.copy(); m=clip(m+sigma*y_w,b)
        # inverse sqrt C
        invsqrt=B@np.diag(1/np.maximum(D,1e-14))@B.T
        ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrt@y_w)
        hsig=float(np.linalg.norm(ps)/np.sqrt(max(1-(1-cs)**(2*max(1,evals//lam)),1e-12))/chi < (1.4+2/(n+1)))
        pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*y_w
        rank_mu=np.zeros((n,n))
        for j,idx in enumerate(sel): rank_mu += ww[j]*np.outer(ys[idx],ys[idx])
        C=(1-c1-cmu)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)+cmu*rank_mu
        C=(C+C.T)/2 + 1e-12*np.eye(n)
        sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chi-1)); sigma=float(np.clip(sigma,1e-9,2*np.max(span)))
    if bestP: return _final(problem,np.asarray(bestP),bestV,evals,first,{},float(evals),0.0,{'host':'CMA-ES'})
    fv,gv=problem.evaluate(m); return _final(problem,np.asarray([m]),[(fv,gv)],evals,first,{},float(evals),0.0,{'host':'CMA-ES'})


def run_dcas_cma(problem, seed=0, pop_size=40, budget=600, gamma=0.97, beta=0.55, cost_kappa=1.0, reward='structural'):
    """DCAS controller embedded in a covariance-adapting Gaussian search host."""
    rng,b,P,vals,farr,feas,ncv,archive,first=_init(problem,seed,pop_size)
    n=problem.dim; span=np.maximum(b[:,1]-b[:,0],1e-12); evals=pop_size
    actions=['cma','wide','narrow','closure','elite']; counts={a:0 for a in actions}; ctrl=DiscountedController(actions,gamma,beta)
    mean=np.mean(P,axis=0); C=np.cov(((P-mean)/span).T)+1e-4*np.eye(n) if pop_size>1 else np.eye(n); sigma=.18
    total_cost=total_gain=0.; recent=[]; asd=[]; rcd=[]
    while evals<budget:
        order=sorted(range(pop_size),key=lambda j:(0,farr[j]) if feas[j] else (1,ncv[j])); elite=P[order[:max(3,pop_size//4)]]; mean=np.mean(elite,axis=0)
        Z=(elite-mean)/span; ifC=np.cov(Z.T)+1e-5*np.eye(n) if len(elite)>1 else np.eye(n)*.05
        C=.8*C+.2*ifC; C=(C+C.T)/2
        eig,B=np.linalg.eigh(C); A=B@np.diag(np.sqrt(np.maximum(eig,1e-8)))
        features=state_features_cached(P,farr,feas,ncv,archive,b,evals,budget,counts,recent)
        for i in range(pop_size):
            if evals>=budget:break
            fi,gi=vals[i]; avail=[a for a in actions if not (a=='closure' and not archive)]; a=ctrl.choose(avail); counts[a]+=1
            z=A@rng.normal(size=n)
            if a=='cma': mutant=mean+sigma*span*z
            elif a=='wide': mutant=mean+min(.45,sigma*1.8)*span*z
            elif a=='narrow': mutant=mean+max(.02,sigma*.45)*span*z
            elif a=='closure':
                anc=nearest_archive(P[i],archive,b); mutant=P[i]+.65*(anc-P[i])+.15*span*z
            else:
                best=P[order[0]]; mutant=P[i]+.55*(best-P[i])+.12*span*z
            trial=clip(mutant,b); ft,gt=problem.evaluate(trial); evals+=1; tf=problem.feasible(gt); tn=problem.ncv(gt)
            if tf:
                archive.append(trial.copy()); archive=archive[-80:]
                if first==0:first=evals
            before=state_description_cached(problem,P[i],fi,gi,P,farr,feas,archive,b,features)
            f2a=farr.copy(); f2a[i]=ft; fm=feas.copy(); fm[i]=tf; nn=ncv.copy(); nn[i]=tn
            f2=state_features_cached(P,f2a,fm,nn,archive,b,evals,budget,counts,recent)
            after=state_description_cached(problem,trial,ft,gt,P,f2a,fm,archive,b,f2); gain=max(0.,before-after)
            move=float(np.linalg.norm((trial-P[i])/span)); base=cost_kappa if a=='closure' else (1.2 if a in ('wide','elite') else 1.0); cost=base+.15*move
            old=(0,fi) if feas[i] else (1,ncv[i]); new=(0,ft) if tf else (1,tn)
            rr=(gain/max(cost,1e-12)) if reward=='structural' else (1. if new<old else 0.)
            ctrl.update(a,rr); total_cost+=cost; total_gain+=gain; recent.append((ncv[i]-tn,(fi-ft)/(1+abs(fi)))); recent=recent[-20:]
            if new<=old:
                P[i]=trial; vals[i]=(ft,gt); farr[i]=ft; feas[i]=tf; ncv[i]=tn
        # simple success-controlled step adaptation
        sigma=float(np.clip(sigma*(.98 if features[0]>.8 else 1.01),.015,.5))
        aa=np.array(list(counts.values()),float); aa/=max(aa.sum(),1); asd.append(float(np.std(aa))); rcd.append(constraint_disagreement(problem,P,vals,archive,b))
    app={'host':'CMA','allocation_sd':float(np.mean(asd[-5:])) if asd else 0.,'rcd':float(np.mean(rcd[-5:])) if rcd else 0.,'action_entropy':action_entropy(counts)}
    return _final(problem,P,vals,evals,first,counts,total_cost,total_gain,app)
