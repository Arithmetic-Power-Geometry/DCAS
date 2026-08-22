from dataclasses import dataclass
import numpy as np

@dataclass
class Result:
    best_f: float
    feasible: bool
    first_feasible: int
    evals: int
    best_x: np.ndarray
    final_ncv: float
    action_counts: dict
    total_declared_cost: float
    total_structural_gain: float
    applicability: dict


def clip(x,b): return np.minimum(np.maximum(x,b[:,0]),b[:,1])

def binomial(target,mutant,cr,rng):
    d=len(target); mask=rng.random(d)<cr; mask[rng.integers(d)]=True; return np.where(mask,mutant,target)

def scalar_key(problem,f,g,mode='ncv',eps=0.0,penalty=100.0):
    feas=problem.feasible(g); v=problem.ncv(g)
    if mode=='pen': return (0,f+penalty*v)
    if mode=='eps': return (0,f) if v<=eps else (1,v)
    if feas: return (0,f)
    if mode=='cv': return (1,problem.cv(g))
    return (1,v)

def archive_distance(x,archive,bounds):
    if len(archive)==0: return None
    span=np.maximum(bounds[:,1]-bounds[:,0],1e-12)
    z=(np.asarray(x)-bounds[:,0])/span
    A=np.asarray([(a-bounds[:,0])/span for a in archive])
    return float(np.linalg.norm(A-z,axis=1).min())

def nearest_archive(x,archive,bounds):
    if len(archive)==0: return None
    span=np.maximum(bounds[:,1]-bounds[:,0],1e-12); z=(x-bounds[:,0])/span
    A=np.asarray(archive); ZA=(A-bounds[:,0])/span
    return A[int(np.argmin(np.linalg.norm(ZA-z,axis=1)))].copy()

def population_diversity(P,bounds):
    span=np.maximum(bounds[:,1]-bounds[:,0],1e-12)
    Z=(P-bounds[:,0])/span
    return float(np.mean(np.std(Z,axis=0)))

def action_entropy(counts):
    a=np.asarray(list(counts.values()),float)
    if a.sum()==0:return 0.0
    p=a/a.sum(); p=p[p>0]
    return float(-np.sum(p*np.log(p)))

class DiscountedController:
    def __init__(self, actions, gamma=0.97, beta=0.55):
        self.actions=list(actions); self.gamma=gamma; self.beta=beta
        self.n={a:0.0 for a in actions}; self.s={a:0.0 for a in actions}; self.s2={a:0.0 for a in actions}
    def choose(self, available=None):
        avail=list(available) if available is not None else self.actions
        if not avail: raise ValueError('at least one action must be available')
        for a in avail:
            if self.n[a]<0.5: return a
        N=max(sum(self.n[a] for a in avail),1.0)
        scores={}
        for a in avail:
            mu=self.s[a]/max(self.n[a],1e-12)
            var=max(self.s2[a]/max(self.n[a],1e-12)-mu*mu,0.0)
            bonus=self.beta*np.sqrt(np.log1p(N)/max(self.n[a],1e-12)) + 0.1*np.sqrt(var)
            scores[a]=mu+bonus
        return max(scores,key=scores.get)
    def update(self,a,r):
        for k in self.actions:
            self.n[k]*=self.gamma; self.s[k]*=self.gamma; self.s2[k]*=self.gamma
        self.n[a]+=1.0; self.s[a]+=float(r); self.s2[a]+=float(r)*float(r)

class StationaryController:
    def __init__(self, actions, beta=0.55):
        self.actions=list(actions); self.beta=beta; self.n={a:0 for a in actions}; self.mean={a:0.0 for a in actions}
    def choose(self, available=None):
        avail=list(available) if available is not None else self.actions
        if not avail: raise ValueError('at least one action must be available')
        for a in avail:
            if self.n[a]==0:return a
        N=sum(self.n[a] for a in avail)
        return max(avail,key=lambda a:self.mean[a]+self.beta*np.sqrt(np.log1p(N)/self.n[a]))
    def update(self,a,r):
        self.n[a]+=1; self.mean[a]+=(r-self.mean[a])/self.n[a]

def state_features(problem,P,vals,archive,bounds,evals,budget,counts,recent):
    feasible=np.array([problem.feasible(g) for _,g in vals],bool)
    ffrac=float(feasible.mean()); ncv=np.array([problem.ncv(g) for _,g in vals])
    medv=float(np.median(ncv)); div=population_diversity(P,bounds)
    if recent:
        vtrend=float(np.mean([z[0] for z in recent[-10:]])); ftrend=float(np.mean([z[1] for z in recent[-10:]]))
    else:vtrend=ftrend=0.0
    coverage=min(1.0,len(archive)/max(len(P),1))
    ent=action_entropy(counts)/np.log(max(len(counts),2))
    return np.array([ffrac,np.tanh(medv),np.tanh(vtrend),np.tanh(ftrend),div,coverage,ent,evals/max(budget,1)],float)

def dynamic_weights(features):
    ffrac,medv,vtrend,ftrend,div,cov,ent,progress=features
    wF=max(0.15,1.0-ffrac+0.35*medv)
    wO=max(0.10,0.25+0.9*ffrac+0.35*progress)
    wU=max(0.05,0.75*(1-progress)+0.2*(1-cov))
    wD=max(0.05,0.5*(1-div)+0.25*(1-ent))
    w=np.array([wF,wO,wU,wD]); return w/w.sum()

def burden_components(problem,x,f,g,P,vals,archive,bounds,features):
    ncv=problem.ncv(g); ad=archive_distance(x,archive,bounds)
    BF=ncv if ad is None else min(ncv,ad)
    feas_fs=[ff for ff,gg in vals if problem.feasible(gg)]
    if problem.feasible(g) and feas_fs:
        best=min(feas_fs); med=np.median(feas_fs); scale=np.median(np.abs(np.asarray(feas_fs)-med))+0.05*abs(best)+1e-9
        BO=max(0.0,(f-best)/scale)
    else: BO=1.0
    BU=max(0.0,1.0-features[5])
    BD=max(0.0,1.0-min(1.0,2.5*features[4]))
    return np.array([np.log1p(10*BF),np.log1p(max(BO,0)),BU,BD])

def state_description(problem,x,f,g,P,vals,archive,bounds,features):
    w=dynamic_weights(features); b=burden_components(problem,x,f,g,P,vals,archive,bounds,features)
    return float(np.dot(w,b))


def state_features_cached(P, farr, feasible_mask, ncv_arr, archive, bounds, evals, budget, counts, recent):
    ffrac=float(np.mean(feasible_mask)); medv=float(np.median(ncv_arr)); div=population_diversity(P,bounds)
    if recent:
        vtrend=float(np.mean([z[0] for z in recent[-10:]])); ftrend=float(np.mean([z[1] for z in recent[-10:]]))
    else: vtrend=ftrend=0.0
    coverage=min(1.0,len(archive)/max(len(P),1))
    ent=action_entropy(counts)/np.log(max(len(counts),2))
    return np.array([ffrac,np.tanh(medv),np.tanh(vtrend),np.tanh(ftrend),div,coverage,ent,evals/max(budget,1)],float)

def state_description_cached(problem,x,f,g,P,farr,feasible_mask,archive,bounds,features):
    ncv=problem.ncv(g); ad=archive_distance(x,archive,bounds)
    BF=ncv if ad is None else min(ncv,ad)
    feas_fs=farr[feasible_mask]
    if problem.feasible(g) and feas_fs.size:
        best=float(np.min(feas_fs)); med=float(np.median(feas_fs)); scale=float(np.median(np.abs(feas_fs-med)))+0.05*abs(best)+1e-9
        BO=max(0.0,(f-best)/scale)
    else: BO=1.0
    BU=max(0.0,1.0-features[5]); BD=max(0.0,1.0-min(1.0,2.5*features[4]))
    b=np.array([np.log1p(10*BF),np.log1p(max(BO,0)),BU,BD])
    return float(np.dot(dynamic_weights(features),b))

def action_cost(action,move,cost_kappa=1.0):
    base={'rand1':1.0,'pbest':1.0,'closure':cost_kappa,'local':cost_kappa,'consensus':1.25}[action]
    return float(base+0.15*move)

def constraint_disagreement(problem,P,vals,archive,bounds):
    if len(archive)<2:return 0.0
    xs=[]; vs=[]; ds=[]
    for x,(f,g) in zip(P,vals):
        if not problem.feasible(g):
            d=archive_distance(x,archive,bounds)
            if d is not None: xs.append(x); vs.append(problem.ncv(g)); ds.append(d)
    if len(vs)<4:return 0.0
    # rank correlation without scipy dependency
    rv=np.argsort(np.argsort(vs)); rd=np.argsort(np.argsort(ds))
    rho=np.corrcoef(rv,rd)[0,1]
    return float(1.0-np.nan_to_num(rho,nan=1.0))

def _final(problem,P,vals,evals,first,counts,total_cost,total_gain,app):
    feas=[(f,i) for i,(f,g) in enumerate(vals) if problem.feasible(g)]
    if feas:
        bf,idx=min(feas); return Result(float(bf),True,first or evals,evals,P[idx].copy(),0.0,counts,total_cost,total_gain,app)
    idx=min(range(len(P)),key=lambda i:problem.ncv(vals[i][1]))
    return Result(float('inf'),False,first or evals+1,evals,P[idx].copy(),problem.ncv(vals[idx][1]),counts,total_cost,total_gain,app)

def run_de(problem,method='ncv',seed=0,pop_size=40,budget=600,F=0.65,CR=0.9):
    rng=np.random.default_rng(seed); b=problem.bounds; d=problem.dim
    P=rng.uniform(b[:,0],b[:,1],size=(pop_size,d)); vals=[problem.evaluate(x) for x in P]; evals=pop_size
    first=next((i+1 for i,(f,g) in enumerate(vals) if problem.feasible(g)),0)
    while evals<budget:
        progress=evals/budget; eps=max(0.0,0.5*(1-progress)**2)
        for i in range(pop_size):
            if evals>=budget: break
            inds=[j for j in range(pop_size) if j!=i]; a,b1,c=rng.choice(inds,3,replace=False)
            trial=clip(binomial(P[i],P[a]+F*(P[b1]-P[c]),CR,rng),b); ft,gt=problem.evaluate(trial); evals+=1
            if problem.feasible(gt) and first==0:first=evals
            if scalar_key(problem,ft,gt,method,eps=eps,penalty=50*(1+10*progress)) <= scalar_key(problem,*vals[i],method,eps=eps,penalty=50*(1+10*progress)):
                P[i]=trial; vals[i]=(ft,gt)
    return _final(problem,P,vals,evals,first,{},float(evals),0.0,{})

def run_jade(problem,seed=0,pop_size=40,budget=600,p=0.1,c=0.1):
    rng=np.random.default_rng(seed); b=problem.bounds; d=problem.dim
    P=rng.uniform(b[:,0],b[:,1],size=(pop_size,d)); vals=[problem.evaluate(x) for x in P]; evals=pop_size
    first=next((i+1 for i,(f,g) in enumerate(vals) if problem.feasible(g)),0); archive=[]; muF=0.5; muCR=0.5
    while evals<budget:
        SF=[]; SCR=[]; order=sorted(range(pop_size),key=lambda j:scalar_key(problem,*vals[j],'ncv')); top=order[:max(2,int(np.ceil(p*pop_size)))]
        for i in range(pop_size):
            if evals>=budget:break
            Fi=np.clip(muF+0.1*np.tan(np.pi*(rng.random()-0.5)),0.05,1.0); CRi=np.clip(rng.normal(muCR,0.1),0,1)
            pb=P[rng.choice(top)]; pool=[j for j in range(pop_size) if j!=i]; r1=rng.choice(pool); r2=rng.choice([j for j in pool if j!=r1])
            xr2=P[r2] if (not archive or rng.random()<0.5) else archive[rng.integers(len(archive))]
            trial=clip(binomial(P[i],P[i]+Fi*(pb-P[i])+Fi*(P[r1]-xr2),CRi,rng),b); ft,gt=problem.evaluate(trial); evals+=1
            if problem.feasible(gt) and first==0:first=evals
            if scalar_key(problem,ft,gt,'ncv') <= scalar_key(problem,*vals[i],'ncv'):
                archive.append(P[i].copy()); archive=archive[-pop_size:]; P[i]=trial; vals[i]=(ft,gt); SF.append(Fi); SCR.append(CRi)
        if SF:
            muF=(1-c)*muF+c*(np.sum(np.square(SF))/np.sum(SF)); muCR=(1-c)*muCR+c*np.mean(SCR)
    return _final(problem,P,vals,evals,first,{},float(evals),0.0,{})

def run_dcas(problem,seed=0,pop_size=40,budget=600,reward='structural',use_closure=True,discounted=True,gamma=0.97,beta=0.55,cost_kappa=1.0):
    rng=np.random.default_rng(seed); bounds=problem.bounds; d=problem.dim; span=np.maximum(bounds[:,1]-bounds[:,0],1e-12)
    P=rng.uniform(bounds[:,0],bounds[:,1],size=(pop_size,d)); vals=[problem.evaluate(x) for x in P]; evals=pop_size
    farr=np.asarray([z[0] for z in vals],float); feasible_mask=np.asarray([problem.feasible(z[1]) for z in vals],bool); ncv_arr=np.asarray([problem.ncv(z[1]) for z in vals],float)
    archive=[P[i].copy() for i in range(pop_size) if feasible_mask[i]]; archive=archive[-80:]
    first=next((i+1 for i,z in enumerate(feasible_mask) if z),0)
    actions=['rand1','pbest','closure','local','consensus']; counts={a:0 for a in actions}
    ctrl=DiscountedController(actions,gamma,beta) if discounted else StationaryController(actions,beta)
    total_cost=0.0; total_gain=0.0; recent=[]; alloc_snap=[]; rcd_snap=[]
    while evals<budget:
        ranked=sorted(range(pop_size),key=lambda j:(0,farr[j]) if feasible_mask[j] else (1,ncv_arr[j])); pbest_pool=ranked[:max(2,int(.2*pop_size))]
        features=state_features_cached(P,farr,feasible_mask,ncv_arr,archive,bounds,evals,budget,counts,recent)
        for i in range(pop_size):
            if evals>=budget:break
            fi,gi=vals[i]
            available=[a for a in actions if not (a=='closure' and (not archive or not use_closure))]
            action=ctrl.choose(available); counts[action]+=1
            inds=[j for j in range(pop_size) if j!=i]; a,b1,cidx=rng.choice(inds,3,replace=False)
            F=0.45+0.35*(1-features[7]); CR=0.75+0.2*features[0]
            if action=='rand1': mutant=P[a]+F*(P[b1]-P[cidx])
            elif action=='pbest':
                pb=P[rng.choice(pbest_pool)]; mutant=P[i]+0.55*(pb-P[i])+F*(P[b1]-P[cidx])
            elif action=='closure':
                anchor=nearest_archive(P[i],archive,bounds); alpha=np.clip(0.35+0.5*(1-features[0]),0.2,0.9)
                mutant=P[i]+alpha*(anchor-P[i])+0.2*(P[b1]-P[cidx])
            elif action=='local':
                best=P[ranked[0]]; sigma=(0.04+0.08*(1-features[7]))*span; mutant=best+rng.normal(0,sigma,d)
            else:
                elite=P[ranked[:max(3,pop_size//4)]].mean(axis=0); mutant=P[i]+0.45*(elite-P[i])+0.3*(P[b1]-P[cidx])
            trial=clip(binomial(P[i],mutant,CR,rng),bounds); ft,gt=problem.evaluate(trial); evals+=1
            trial_feas=problem.feasible(gt); trial_ncv=problem.ncv(gt)
            if trial_feas:
                archive.append(trial.copy()); archive=archive[-80:]
                if first==0:first=evals
            before=state_description_cached(problem,P[i],fi,gi,P,farr,feasible_mask,archive,bounds,features)
            tmp_farr=farr.copy(); tmp_farr[i]=ft
            tmp_feas=feasible_mask.copy(); tmp_feas[i]=trial_feas
            tmp_ncv=ncv_arr.copy(); tmp_ncv[i]=trial_ncv
            f2=state_features_cached(P,tmp_farr,tmp_feas,tmp_ncv,archive,bounds,evals,budget,counts,recent)
            after=state_description_cached(problem,trial,ft,gt,P,tmp_farr,tmp_feas,archive,bounds,f2)
            gain=max(0.0,before-after); move=float(np.linalg.norm((trial-P[i])/span)); cost=action_cost(action,move,cost_kappa)
            old_key=(0,fi) if feasible_mask[i] else (1,ncv_arr[i]); new_key=(0,ft) if trial_feas else (1,trial_ncv)
            if reward=='structural': r=gain/max(cost,1e-12)
            elif reward=='success': r=1.0 if new_key < old_key else 0.0
            elif reward=='violation': r=max(0.0,ncv_arr[i]-trial_ncv)/max(cost,1e-12)
            elif reward=='objective': r=(max(0.0,fi-ft)/(1+abs(fi)))/max(cost,1e-12) if feasible_mask[i] and trial_feas else 0.0
            else: raise ValueError(reward)
            ctrl.update(action,r); total_cost+=cost; total_gain+=gain
            recent.append((ncv_arr[i]-trial_ncv, (fi-ft)/(1+abs(fi)) if np.isfinite(fi) else 0.0)); recent=recent[-20:]
            if new_key <= old_key:
                P[i]=trial; vals[i]=(ft,gt); farr[i]=ft; feasible_mask[i]=trial_feas; ncv_arr[i]=trial_ncv
        alloc=np.array(list(counts.values()),float); alloc=alloc/max(alloc.sum(),1)
        alloc_snap.append(float(np.std(alloc))); rcd_snap.append(constraint_disagreement(problem,P,vals,archive,bounds))
    app={'allocation_sd':float(np.mean(alloc_snap[-5:])) if alloc_snap else 0.0,'rcd':float(np.mean(rcd_snap[-5:])) if rcd_snap else 0.0,'action_entropy':action_entropy(counts)}
    return _final(problem,P,vals,evals,first,counts,total_cost,total_gain,app)

