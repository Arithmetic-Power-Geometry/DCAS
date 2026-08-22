import os, sys
import pandas as pd
import streamlit as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','src'))
from dcas import (problem_by_name, all_problems, pressure_vessel, tension_spring, cantilever_beam,
                  run_dcas, run_de, run_jade, run_cmaes, run_pso, run_dcas_cma, run_dcas_pso)

st.set_page_config(page_title='DCAS Explorer',layout='wide')
st.title('DCAS Explorer')
st.caption('Interactive laboratory for Discovery-Closure Adaptive Search: principal, cross-host, engineering, and preserved stress-test evidence.')

extra=[pressure_vessel(),tension_spring(),cantilever_beam()]
problems={p.name:p for p in all_problems()+extra}
methods=['DCAS-DE','UCB-Success','UCB-Violation','UCB-Objective','DCAS-noClosure','DCAS-stationary','JADE-NCV','DE-NCV','DE-EPS','DE-PEN','CMA-ES','DCAS-CMA','PSO','DCAS-PSO']
with st.sidebar:
    st.header('Experiment controls')
    problem=st.selectbox('Problem',sorted(problems))
    method=st.selectbox('Method',methods)
    pop_size=st.slider('Population size',10,100,40,5)
    budget=st.slider('Evaluation budget',100,20000,400,100)
    seed=st.number_input('Seed',0,100000,0)
    kappa=st.select_slider('Closure/local declared cost multiplier',options=[1,2,5,10,25,50,100],value=1)
    gamma=st.slider('Discount factor gamma',0.80,0.999,0.97,0.001)
    beta=st.slider('Exploration beta',0.0,2.0,0.55,0.05)
    run=st.button('Run experiment',type='primary')

def execute(method,p,seed,budget,pop_size,kappa,gamma,beta):
    kw=dict(seed=int(seed),budget=int(budget),pop_size=int(pop_size))
    if method=='DCAS-DE': return run_dcas(p,**kw,reward='structural',gamma=gamma,beta=beta,cost_kappa=kappa)
    if method=='UCB-Success': return run_dcas(p,**kw,reward='success',gamma=gamma,beta=beta,cost_kappa=kappa)
    if method=='UCB-Violation': return run_dcas(p,**kw,reward='violation',gamma=gamma,beta=beta,cost_kappa=kappa)
    if method=='UCB-Objective': return run_dcas(p,**kw,reward='objective',gamma=gamma,beta=beta,cost_kappa=kappa)
    if method=='DCAS-noClosure': return run_dcas(p,**kw,reward='structural',use_closure=False,gamma=gamma,beta=beta,cost_kappa=kappa)
    if method=='DCAS-stationary': return run_dcas(p,**kw,reward='structural',discounted=False,beta=beta,cost_kappa=kappa)
    if method=='JADE-NCV': return run_jade(p,**kw)
    if method in ['DE-NCV','DE-EPS','DE-PEN']: return run_de(p,**kw,method={'DE-NCV':'ncv','DE-EPS':'eps','DE-PEN':'pen'}[method])
    if method=='CMA-ES': return run_cmaes(p,**kw)
    if method=='DCAS-CMA': return run_dcas_cma(p,**kw,gamma=gamma,beta=beta,cost_kappa=kappa)
    if method=='PSO': return run_pso(p,**kw)
    if method=='DCAS-PSO': return run_dcas_pso(p,**kw,gamma=gamma,beta=beta,cost_kappa=kappa)
    raise KeyError(method)

single,preserved,cross,large,external_tab,sweep=st.tabs(['Single run','Main 3,600-run campaign','Cross-host 1,440-run campaign','Large-scale stress','External CEC/SOTA','Parameter sweep'])
with single:
    if run:
        p=problems[problem]; r=execute(method,p,seed,budget,pop_size,kappa,gamma,beta)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric('Feasible',str(r.feasible)); c2.metric('First feasible',r.first_feasible)
        c3.metric('Best objective',f'{r.best_f:.6g}' if r.feasible else 'n/a')
        c4.metric('Final NCV',f'{r.final_ncv:.4g}'); c5.metric('Declared cost',f'{r.total_declared_cost:.2f}')
        if r.applicability: st.json(r.applicability)
        if r.action_counts: st.bar_chart(pd.Series(r.action_counts,name='count'))
        st.code(', '.join(f'{v:.6g}' for v in r.best_x))

def table_tab(path, caption):
    if os.path.exists(path):
        df=pd.read_csv(path); st.caption(caption+f' Rows: {len(df):,}.')
        cols=st.columns(2); probs=cols[0].multiselect('Problems',sorted(df.problem.unique()),default=sorted(df.problem.unique()),key=path+'p'); meths=cols[1].multiselect('Methods',sorted(df.method.unique()),default=sorted(df.method.unique()),key=path+'m')
        sub=df[df.problem.isin(probs)&df.method.isin(meths)]; st.dataframe(sub,use_container_width=True,height=410); st.download_button('Download filtered CSV',sub.to_csv(index=False),os.path.basename(path),key=path+'d')
    else: st.info('Dataset not found in this checkout.')
base=os.path.join(os.path.dirname(__file__),'..','data','raw')
with preserved: table_tab(os.path.join(base,'results.csv'),'Corrected 30-seed principal campaign.')
with cross: table_tab(os.path.join(base,'crosshost.csv'),'Cross-host/engineering campaign: DCAS-DE, DCAS-CMA, DCAS-PSO and their host baselines.')
with large: table_tab(os.path.join(base,'large_scale.csv'),'Scalable 100/500/1000-dimensional stress campaign.')
with external_tab:
    ep=os.path.join(os.path.dirname(__file__),'..','data','external','results.csv')
    if os.path.exists(ep):
        edf=pd.read_csv(ep); st.success(f'Executed external result file detected: {len(edf):,} rows.'); st.dataframe(edf,use_container_width=True,height=420)
    else:
        st.info('No executed official external result CSV is present in this checkout. This is intentional: missing author-code/official-evaluator runs are never fabricated.')
    st.markdown('Use **GitHub Actions -> External CEC-SOTA Validation** to fetch verified public sources, build the frozen job manifest, run preflight, and analyze any standardized external results.')
    up=st.file_uploader('Inspect a standardized external result CSV',type=['csv'],key='external-upload')
    if up is not None:
        udf=pd.read_csv(up); st.dataframe(udf,use_container_width=True,height=360); st.download_button('Download uploaded CSV',udf.to_csv(index=False),'external_results_checked.csv')
with sweep:
    st.write('Interactive sweeps are capped to keep the application responsive.')
    sweep_param=st.selectbox('Sweep parameter',['gamma','beta','kappa'])
    values={'gamma':[0.85,0.90,0.95,0.97,0.99],'beta':[0.0,0.25,0.55,1.0,1.5],'kappa':[1,2,5,10,50]}[sweep_param]
    sweep_seeds=st.slider('Seeds per value',1,5,2)
    if st.button('Run parameter sweep'):
        p=problems[problem]; rows=[]; prog=st.progress(0); total=len(values)*sweep_seeds; done=0
        for val in values:
            for sd in range(sweep_seeds):
                gg,bb,kk=gamma,beta,kappa
                if sweep_param=='gamma':gg=val
                elif sweep_param=='beta':bb=val
                else:kk=val
                r=execute('DCAS-DE',p,sd,budget,pop_size,kk,gg,bb)
                rows.append({sweep_param:val,'seed':sd,'feasible':r.feasible,'first_feasible':r.first_feasible,'best_f':r.best_f,'final_ncv':r.final_ncv,'declared_cost':r.total_declared_cost}); done+=1; prog.progress(done/total)
        sdf=pd.DataFrame(rows); st.dataframe(sdf,use_container_width=True); st.line_chart(sdf.groupby(sweep_param)['first_feasible'].median())
