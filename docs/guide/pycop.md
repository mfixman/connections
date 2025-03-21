# PyCoP Provers

The Connections package includes standalone connection provers for classical, intuitionistic, and modal logics. These provers are Python implementations of the leanCoP, ileanCoP, and MleanCoP provers.

## Available Provers

- **pyCoP**: Classical first-order logic prover
- **ipyCoP**: Intuitionistic first-order logic prover
- **mpyCoP**: Modal logic prover (S4, S5, D, T)

## Command Line Usage

The provers can be invoked from the command line:

```bash
python pycop.py <file> [logic] [domain]
```

### Arguments

- `<file>`: Path to the problem file in TPTP/QMLTP syntax
- `[logic]`: Optional logic type (default: "classical")
  - `"classical"`: Classical logic
  - `"intuitionistic"`: Intuitionistic logic
  - `"D"`: Modal logic D
  - `"T"`: Modal logic T
  - `"S4"`: Modal logic S4
  - `"S5"`: Modal logic S5
- `[domain]`: Optional domain type for modal logics (default: "constant")
  - `"constant"`: Constant domain
  - `"cumulative"`: Cumulative domain
  - `"varying"`: Varying domain

### Examples

Classical logic:
```bash
python pycop.py tests/cnf_problems/SYN001+1.p
```

Intuitionistic logic:
```bash
python pycop.py tests/cnf_problems/SYN001+1.p intuitionistic
```

Modal logic S4 with varying domain:
```bash
python pycop.py tests/cnf_problems/SYN001+1.p S4 varying
```

## Python API

The provers can also be used programmatically:

```python
from connections.provers import ClassicalProver, IntuitionisticProver, ModalProver

# Classical prover
prover = ClassicalProver()
result = prover.prove("problem_file.p")
print(result.status)  # "Theorem" if proven

# Intuitionistic prover
prover = IntuitionisticProver()
result = prover.prove("problem_file.p")

# Modal prover
prover = ModalProver(logic="S4", domain="constant")
result = prover.prove("problem_file.p")
```

## Comparison with Original Provers

These provers are equivalent to version 1.0f of:
- leanCoP (classical logic)
- ileanCoP (intuitionistic logic)
- MleanCoP (modal logics)

The original Prolog implementations can be found in the `comparisons` directory.

## Performance Notes

- The provers use a connection-driven search strategy
- Backtracking is restricted using regularity and lemmata
- For modal logics, prefixes are used to handle the accessibility relation
- The implementation is optimized for Python while maintaining the original algorithms

## Example Output

When a theorem is proven, the output includes:
- Proof status ("Theorem" or "Non-theorem")
- Proof sequence (if found)
- Time taken
- Search statistics

Example:
```python
from connections.provers import ClassicalProver

prover = ClassicalProver()
result = prover.prove("tests/cnf_problems/SYN001+1.p")

print(result.status)  # "Theorem"
print(result.proof_sequence)  # List of proof steps
print(result.statistics)  # Search statistics
``` 