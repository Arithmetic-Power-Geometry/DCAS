from pathlib import Path
import csv, json, sys

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'external'
DATA = ROOT / 'data' / 'external'
DATA.mkdir(parents=True, exist_ok=True)

# Fail-closed: numerical jobs are generated only from protocol metadata resolved
# from the official/released package after source retrieval and adapter preflight.
resolved = EXT / 'protocols_resolved.json'
plan = [
    {'suite':'CEC2010-CSOP','status':'protocol_pending','target':'released/official evaluator'},
    {'suite':'CEC2017-CSOP','status':'source_verified_protocol_pending','target':'official CEC evaluator'},
    {'suite':'CEC2020-CSOP','status':'protocol_pending','target':'released/official evaluator'},
    {'suite':'CEC2025-BC-CSOP','status':'source_verified_protocol_pending','target':'RDEx-CSOP + official track'},
]
with (DATA/'external_campaign_plan.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=plan[0].keys()); w.writeheader(); w.writerows(plan)

if not resolved.exists():
    # Empty manifest is intentional: no guessed run counts/dimensions/budgets.
    (DATA/'job_manifest.csv').write_text('suite,problem,dimension,seed,eval_budget\n')
    summary={'jobs':0,'policy':'fail-closed','status':'protocols_unresolved',
             'note':'Run source fetch + adapter protocol resolution before numerical execution.'}
    (DATA/'job_manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    sys.exit(0)

protocols=json.loads(resolved.read_text())
rows=[]
for spec in protocols.get('suites',[]):
    required=('suite','problems','dimensions','runs','eval_budget')
    if any(k not in spec for k in required):
        raise SystemExit(f'Incomplete resolved protocol: {spec}')
    for d in spec['dimensions']:
        budget = spec['eval_budget'] * d if spec.get('budget_is_multiplier',False) else spec['eval_budget']
        for prob in spec['problems']:
            for seed in range(1,int(spec['runs'])+1):
                rows.append({'suite':spec['suite'],'problem':prob,'dimension':d,'seed':seed,'eval_budget':budget})

out=DATA/'job_manifest.csv'
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','problem','dimension','seed','eval_budget'])
    w.writeheader(); w.writerows(rows)
summary={'jobs':len(rows),'policy':'resolved-official-protocols','status':'ready' if rows else 'empty'}
(DATA/'job_manifest.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
