from pathlib import Path
import runpy
import sys

folder = Path(__file__).resolve().parent / 'packages/axiom_prediction/experiments'
sys.path.insert(0, str(folder))
runpy.run_path(str(folder / 'wednesday_monitor.py'), run_name = '__main__')
