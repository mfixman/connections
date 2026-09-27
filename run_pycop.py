from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "src"), str(root / "packages/pycop/src")]

if __name__ == "__main__":
    from pycop.run_pycop import main

    raise SystemExit(main())
