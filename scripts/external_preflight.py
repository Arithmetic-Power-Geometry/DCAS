from pathlib import Path
import json, subprocess, csv
ROOT=Path(__file__).resolve().parents[1]
sources=json.loads((ROOT/'external/sources.json').read_text())
report={'sources':{},'protocols_resolved':False,'manifest':{},'ready_for_numeric_campaign':True}
for name,s in sources.items():
    p=ROOT/s['path']
    item={'path':str(p.relative_to(ROOT)),'exists':p.exists(),'required':s['required'],'role':s['role']}
    if p.exists() and (p/'.git').exists():
        try: item['commit']=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        except Exception: item['commit']='unknown'
    report['sources'][name]=item
    if s['required'] and not p.exists(): report['ready_for_numeric_campaign']=False
resolved=ROOT/'external/protocols_resolved.json'
report['protocols_resolved']=resolved.exists()
if not resolved.exists(): report['ready_for_numeric_campaign']=False
mf=ROOT/'data/external/job_manifest.csv'
if mf.exists():
    with mf.open() as f: report['manifest']['jobs']=sum(1 for _ in csv.DictReader(f))
else:
    report['manifest']['jobs']=0
if report['manifest']['jobs']==0: report['ready_for_numeric_campaign']=False
rd=ROOT/'external/vendor/RDEx-Series/RDEx_CSOP'; cec=ROOT/'external/vendor/CEC2017'
report['rdex_csop_package_found']=rd.exists(); report['cec2017_package_found']=cec.exists()
out=ROOT/'data/external/preflight.json'; out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
