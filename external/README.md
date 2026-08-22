# External CEC and contemporary-SOTA validation

The frozen DCAS implementation is not changed by this layer.  This directory keeps third-party source code outside the Apache-2.0 project and records provenance before any external numerical result can enter the manuscript.

## Verified public sources

* RDEx series / RDEx-CSOP: `https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series`
* Official constrained CEC 2017 repository: `https://github.com/P-N-Suganthan/CEC2017`
* CMA-ES reference implementation: `https://github.com/CMA-ES/pycma`

The RDEx repository reports Rank-1 U-score for its CEC-2025 BC-CSOP package.  Its package includes algorithm source and benchmark material.  Third-party licenses remain upstream and are never relicensed as Apache-2.0 here.

## Fail-closed rule

No external score is reported merely because a repository was downloaded.  A result enters `data/external/results.csv` only after: (1) source provenance is recorded, (2) the benchmark adapter passes its self-test, (3) the official budget and dimensions are checked, and (4) the result satisfies the common schema in `external/RESULT_SCHEMA.md`.

## One-click GitHub workflow

Run **Actions -> External CEC/SOTA Validation -> Run workflow**.  The workflow fetches verified public repositories, creates the frozen job manifest, performs provenance/preflight checks, and analyzes any standardized result file produced by enabled adapters.  Missing or unverified author-code sources are reported as `not executed`, never replaced by a local imitation.

## Local commands

```bash
bash scripts/fetch_external_sota.sh
python scripts/prepare_external_campaign.py
python scripts/external_preflight.py
# after official/author runners produce data/external/results.csv:
python scripts/analyze_external_results.py
```

## Bibliographic provenance note

DRL-AEOSF is the 2026 *Swarm and Evolutionary Computation* article by Eryang Guo, Yuelin Gao, Chenyang Gao, and Mengqi Jiang (DOI `10.1016/j.swevo.2026.102453`). No GitHub URL is inserted for DRL-AEOSF unless an author-verifiable public implementation is located.

Competition-specific run counts, dimensions, tolerances, budgets, and scoring rules always override generic manifest defaults. The workflow must encode or read the official protocol before results can be admitted.
