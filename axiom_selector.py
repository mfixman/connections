from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "src"), str(root / "packages")]

if __name__ == "__main__":
    from axiom_prediction.cli import main

    raise SystemExit(main())
