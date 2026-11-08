# Default HPC locations of the axiom-selection goal; every script takes them as command-line options.
from pathlib import Path
import sys

home = Path('/home/msf43')
goal = home / 'axiom-goal'
connections = home / 'connections'
tptp = Path('/rds/user/msf43/hpc-work/TPTP')
big_train = connections / 'results/big-train'
prover9_selection = home / 'TP/Prover9/axiom_selection'
eprover = home / 'TP/eprover/bin/eprover'

def add_connections(checkout = connections):
    # The scripts run outside the package, so the checkout's packages and src go first on sys.path.
    checkout = Path(checkout)
    sys.path[:0] = [str(checkout / 'packages'), str(checkout / 'src')]

def add_prover9(directory = prover9_selection):
    sys.path.insert(0, str(directory))
