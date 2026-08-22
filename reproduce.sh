#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
python scripts/run_benchmark.py --seeds 30 --budget 400 --jobs 2
python scripts/run_cost_study.py
python scripts/make_artifacts.py
python scripts/run_crosshost.py --seeds 30 --budget 400 --jobs 2 --problems 'G06,G08,RotatedBox10,NarrowCorridor12,WeldedBeam,PressureVessel,TensionSpring,CantileverBeam'
python scripts/run_large_scale.py --dims 100,500,1000 --seeds 10 --budget 600 --jobs 2
python scripts/analyze_extended.py
