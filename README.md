# Discovery-Closure Adaptive Search (DCAS)

DCAS is a reproducible research implementation of **cost-normalized structural resource allocation** for constrained black-box optimization. The implementation uses **availability-gated action selection**: an action can receive credit only when it is executable and actually executed. In particular, closure is temporarily excluded while the feasible archive is empty.

## Repository

Official repository: https://github.com/Arithmetic-Power-Geometry/DCAS

```bash
git clone https://github.com/Arithmetic-Power-Geometry/DCAS.git
cd DCAS
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The repository uses a `src/` layout. Pytest is configured to resolve `src` directly, while installation with `pip install -e .` remains the recommended development workflow.

## One-click GitHub workflow
Upload this folder to a GitHub repository and open **Actions -> reproduce-dcas -> Run workflow**. The default paper profile uses 30 independent seeds and a 400-evaluation budget, runs the test suite, regenerates the full main benchmark, runs the declared-cost study, rebuilds every table and figure, and uploads the results artifact.

## Local reproduction
```bash
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_benchmark.py --seeds 30 --budget 400 --jobs 8
python scripts/run_cost_study.py
python scripts/make_artifacts.py
```

## Interactive application
```bash
streamlit run app/app.py
```
The app exposes problem, method, population size, evaluation budget, random seed, cost multiplier, discount factor and exploration coefficient. It supports single runs, browsing/downloading the complete preserved paper results, and small parameter sweeps.

## Corrected paper campaign
The bundled main campaign contains **3,600 real runs**: 12 constrained problems x 10 algorithms/ablations x 30 independent seeds, all under a common 400-evaluation budget. The separate declared-cost mechanism study contains **108 corrected runs**. All paper tables and figures are generated from these preserved CSV files.

## Reproducibility boundary
The repository contains directly executed DCAS, matched reward/controller ablations, JADE-NCV and classical DE constraint-handling baselines. It does not pretend to reimplement author-released RDEx-CSOP or DRL-AEOSF. Those methods are treated as external state-of-the-art benchmarks for the next official CEC campaign.

## License
Apache License 2.0. See `LICENSE`.


## Official CEC / contemporary-SOTA layer

The repository now includes a fail-closed external validation layer. `scripts/prepare_external_campaign.py` freezes the job manifest; `scripts/fetch_external_sota.sh` retrieves verified public RDEx-CSOP, CEC2017 and pycma sources in GitHub/networked environments; `scripts/external_preflight.py` records commits and evaluator availability; and `scripts/analyze_external_results.py` produces common ranks, Holm-corrected pairwise tests, A12, ECDF and performance profiles from executed standardized results. No unexecuted external score is included in the paper.
