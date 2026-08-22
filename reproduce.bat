@echo off
setlocal
python -m pip install -r requirements.txt || exit /b 1
python -m pip install -e . || exit /b 1
python -m pytest -q || exit /b 1
python scripts\run_benchmark.py --seeds 30 --budget 400 --jobs 2 || exit /b 1
python scripts\run_cost_study.py || exit /b 1
python scripts\make_artifacts.py || exit /b 1
python scripts\run_crosshost.py --seeds 30 --budget 400 --jobs 2 --problems "G06,G08,RotatedBox10,NarrowCorridor12,WeldedBeam,PressureVessel,TensionSpring,CantileverBeam" || exit /b 1
python scripts\run_large_scale.py --dims 100,500,1000 --seeds 10 --budget 600 --jobs 2 || exit /b 1
python scripts\analyze_extended.py || exit /b 1
echo DCAS reproduction completed successfully.
endlocal
