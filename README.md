# MyPyCoP

Small research fork of the `connections` prover codebase, trimmed to the parts
used for running:

```bash
python pycop.py ...
python compare_strategies.py ...
```

The retained runtime supports three calculi only:

- `Classical`
- `ClassicalCut`
- `ClassicalSAT`

It keeps the strategy code used to compare those calculi, plus classical TPTP
input handling and the classical Prolog translator in `translation/classical/`.

## Requirements

- Python 3.10 or later
- A C++ compiler and CMake to build the pinned `pydical` fork
- SWI-Prolog 8.4.3 for classical FOF-to-CNF translation

## Installation

```bash
pip install .
```

This installs the pinned `pydical` revision used by the `ClassicalSAT` code.

## Entry Points

Run a single strategy on one problem:

```bash
python pycop.py tests/cnf_problems/SYN001+1.p --strategy FirstAction
python pycop.py tests/cnf_problems/SYN001+1.p --strategy LeanCopCut
python pycop.py tests/cnf_problems/SYN001+1.p --strategy SATSolve
```

Compare several strategies across a directory or TPTP slice:

```bash
python compare_strategies.py tests/cnf_problems --strategies FirstAction LeanCopCut --csv results.csv
python compare_strategies.py tests/cnf_problems --strategies SATControl SATTest --csv results.csv
```

Both entrypoints accept:

- local `.p` / `.ax` files
- directories of classical TPTP problems
- TPTP codenames such as `SYN001+1`
- raw quoted TPTP problem text
- already-translated connection CNF files

## Minimal API Example

```python
from connections.env import ConnectionEnv, Settings

env = ConnectionEnv("tests/cnf_problems/SYN001+1.p", settings=Settings(logic="Classical"))
observation = env.reset()

while True:
    action = env.action_space[0]
    observation, reward, done, info = env.step(action)
    if done:
        print(info)
        break
```

## Translation

Raw classical FOF problems are translated through:

```bash
translation/classical/translate.sh <file> <output>
```

The repository keeps only the classical translation pipeline.

## Citation

```
@inproceedings{connections_2023,
    author     = {Rømming, Fredrik and Otten, Jens and Holden, Sean B.},
    title      = {Connections: {Markov} {Decision} {Processes} for {Classical}, 
                  {Intuitionistic} and {Modal} {Connection} {Calculi}},
    booktitle  = {Proceedings of the 1st {International} {Workshop} on    
                  {Automated} {Reasoning} with {Connection} {Calculi}},
    series     = {{CEUR} {Workshop} {Proceedings}},
    volume     = {3613},
    year       = {2023},
    pages      = {107--118},
}
```
