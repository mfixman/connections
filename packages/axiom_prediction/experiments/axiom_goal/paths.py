# Default HPC locations of the axiom-selection goal; every script takes them as command-line options.
from pathlib import Path
import sys

home = Path('/home/msf43')
goal = home / 'axiom-goal'
connections = Path(__file__).resolve().parents[4]
external = connections / 'external'
tptp = Path('/rds/user/msf43/hpc-work/TPTP')
big_train = connections / 'results/big-train'
prover9_selection = external / 'Prover9/axiom_selection'
eprover = external / 'eprover/bin/eprover'
spass = external / 'SPASS/SPASS'
iprover = external / 'iprover/iproveropt'

def add_connections(checkout = connections):
    # The scripts run outside the package, so the checkout's packages and src go first on sys.path.
    checkout = Path(checkout)
    sys.path[:0] = [str(checkout / 'packages'), str(checkout / 'src')]

def add_prover9(directory = prover9_selection):
    sys.path.insert(0, str(directory))
