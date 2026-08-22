# External result schema

Every executed run must contribute one row to `data/external/results.csv` with:

`suite, problem, dimension, seed, method, eval_budget, success, final_violation, best_f, first_feasible, runtime_s, source_commit, evaluator_id`

Optional trace files go to `data/external/traces/`.  `success` is 0/1.  `first_feasible` equals `eval_budget+1` if feasibility was never reached.  No objective values are averaged across different functions.  Ranking is performed within suite/problem/dimension/seed blocks using feasibility first, then violation for infeasible runs, and objective/EFF for feasible runs.
