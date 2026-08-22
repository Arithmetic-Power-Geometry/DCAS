# Discovery-Closure Adaptive Search (DCAS)

DCAS is a reproducible research implementation of **cost-normalized structural resource allocation** for constrained black-box optimization. The implementation uses **availability-gated action selection**: an action can receive credit only when it is executable and actually executed. In particular, closure is temporarily excluded while the feasible archive is empty.

## Paper

Akhtar, M. A. K. (2026). *Discovery–Closure Adaptive Search: Cost-Normalized Structural Resource Allocation for Constrained Black-Box Optimization* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22058689

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

## One-Click GitHub Workflow

Upload this folder to a GitHub repository and open **Actions → reproduce-dcas → Run workflow**. The default paper profile uses 30 independent seeds and a 400-evaluation budget, runs the test suite, regenerates the full main benchmark, runs the declared-cost study, rebuilds every empirical table and result figure, and uploads the results artifact.

## Local Reproduction

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_benchmark.py --seeds 30 --budget 400 --jobs 8
python scripts/run_cost_study.py
python scripts/make_artifacts.py
```

## Interactive Application

```bash
streamlit run app/app.py
```

The application exposes benchmark problem, method, population size, evaluation budget, random seed, declared cost multiplier, discount factor, and exploration coefficient. It supports single runs, browsing and downloading the complete preserved paper results, and capped interactive parameter sweeps without editing source code.

## Corrected Paper Campaign

The bundled principal campaign contains **3,600 executed runs**:

- 12 constrained problems
- 10 algorithms/ablations
- 30 independent seeds
- 400 evaluations per run

The separate declared-cost mechanism study contains **108 runs**.

The complete preserved experimental record consists of:

- **3,600** principal runs
- **108** declared-cost runs
- **1,440** cross-host/engineering runs
- **240** large-scale stress runs
- **5,388 total preserved records**

All empirical result tables and result figures reported in the paper are generated from the preserved CSV files.

## Cross-Host Validation

DCAS is evaluated through multiple host realizations:

- **DCAS-DE**
- **DCAS-CMA**
- **DCAS-PSO**

The cross-host experiments test whether the structural resource-allocation principle transfers beyond a single evolutionary search mechanism. The preserved results retain both positive and negative findings rather than reporting only favorable host combinations.

## Reproducibility Boundary

The repository contains directly executed DCAS implementations, matched reward/controller ablations, JADE-NCV, and classical DE constraint-handling baselines.

It does **not** present reconstructed implementations of author-released RDEx-CSOP or DRL-AEOSF as direct experimental evidence. These methods are reserved for external validation using verified author code and common official evaluation protocols.

No unexecuted external score is reported as an empirical result in the paper.

## Official CEC / Contemporary-SOTA Layer

The repository includes a fail-closed external-validation infrastructure for future official CEC and contemporary state-of-the-art comparisons.

`scripts/prepare_external_campaign.py` freezes the experimental job manifest.

`scripts/fetch_external_sota.sh` retrieves verified public RDEx-CSOP, CEC2017, and pycma sources in GitHub/networked environments.

`scripts/external_preflight.py` records source commits and evaluator availability before execution.

`scripts/analyze_external_results.py` provides standardized analysis of executed external results, including:

- common ranks
- Holm-corrected pairwise tests
- Vargha–Delaney A12 effect sizes
- ECDF analysis
- performance profiles

The external-validation pipeline is intentionally fail-closed: unavailable or unexecuted third-party methods cannot silently produce benchmark scores.

Third-party software and benchmark licenses remain upstream and separate from the DCAS Apache-2.0 license.

## Reproducibility

The package provides:

- complete Python source code
- fixed experimental seeds
- preserved raw CSV results
- generated empirical tables and figures
- unit and regression tests
- action-availability regression testing
- budget-completion tests
- GitHub Actions workflows
- interactive Streamlit application
- external-validation preflight checks
- standardized external-result schema
- provenance tracking

These components are intended to make the reported experiments independently inspectable and reproducible.

## Citation

If you use DCAS or the accompanying research, please cite:

**APA**

Akhtar, M. A. K. (2026). *Discovery–Closure Adaptive Search: Cost-Normalized Structural Resource Allocation for Constrained Black-Box Optimization* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22058689

**BibTeX**

```bibtex
@misc{akhtar2026dcas,
  author    = {Akhtar, Mohammad Amir Khusru},
  title     = {Discovery--Closure Adaptive Search: Cost-Normalized Structural Resource Allocation for Constrained Black-Box Optimization},
  year      = {2026},
  publisher = {Zenodo},
  version   = {V1},
  doi       = {10.5281/zenodo.22058689},
  url       = {https://doi.org/10.5281/zenodo.22058689}
}
```

## License

DCAS is released under the **Apache License 2.0**. See `LICENSE` for details.
