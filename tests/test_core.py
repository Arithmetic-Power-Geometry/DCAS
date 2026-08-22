import numpy as np
from dcas import all_problems, run_dcas, run_de, run_jade
from dcas.core import archive_distance, DiscountedController, dynamic_weights

def test_archive_distance_zero():
    b=np.array([[0,1],[0,1]],float); x=np.array([.2,.3]); assert archive_distance(x,[x.copy()],b)==0.0

def test_dcas_budget_and_actions():
    p=all_problems()[1]; r=run_dcas(p,seed=1,budget=120,pop_size=20); assert r.evals==120; assert sum(r.action_counts.values())>0

def test_baselines_budget():
    p=all_problems()[0]; assert run_de(p,seed=1,budget=100,pop_size=20).evals==100; assert run_jade(p,seed=1,budget=100,pop_size=20).evals==100

def test_discounted_controller_explores():
    c=DiscountedController(['a','b'],gamma=.9,beta=.5); a1=c.choose(); c.update(a1,1.0); a2=c.choose(); assert a2!=a1

def test_weights_sum_one():
    w=dynamic_weights(np.array([.2,.3,0,0,.1,.2,.5,.4])); assert abs(w.sum()-1)<1e-12 and np.all(w>0)


def test_unavailable_closure_does_not_starve_other_actions():
    c=DiscountedController(['rand1','pbest','closure','local','consensus'],gamma=.97,beta=.55)
    available=['rand1','pbest','local','consensus']
    chosen=[]
    for _ in range(4):
        a=c.choose(available); chosen.append(a); c.update(a,0.0)
    assert set(chosen)==set(available)
    assert c.n['closure']==0.0

def test_crosshost_cma_budget():
    from dcas import run_cmaes, run_dcas_cma
    p=all_problems()[1]
    assert run_cmaes(p,seed=2,budget=100,pop_size=20).evals==100
    assert run_dcas_cma(p,seed=2,budget=100,pop_size=20).evals==100

def test_crosshost_pso_budget():
    from dcas import run_pso, run_dcas_pso
    p=all_problems()[1]
    assert run_pso(p,seed=3,budget=100,pop_size=20).evals==100
    assert run_dcas_pso(p,seed=3,budget=100,pop_size=20).evals==100


def test_external_sources_are_fail_closed():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    src=json.loads((root/'external/sources.json').read_text())
    assert src['rdex_csop']['required'] is True
    assert src['cec2017_official']['required'] is True
    assert src['drl_aeosf']['url']==''

def test_external_result_schema_documented():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    t=(root/'external/RESULT_SCHEMA.md').read_text()
    for c in ['suite','problem','dimension','seed','method','source_commit','evaluator_id']:
        assert c in t
