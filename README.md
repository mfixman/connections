# SATResetCoP

`SATResetCoP` is a strategy in the connections prover that aggressively resets the tableau instead of backtracking.

This repo contains two Python files that can be run standalone.

`pycop.py` gets as input a single strategy and a single `.p` file, in either FOF or CNF format.
This can be expressed as the path to this file, or a TPTP file specification; in the latter case, the file's path is expected to be inside where the main TPTP directory, which can be specified either with the `--tptp` option or with the `\$TPTP` environment variable points.

```
$ python pycop.py --strategy SATResetCoP --print-trace --debug PUZ020-1.p
Start st1: [person(wife)]
Extension ex3: person(wife) -> [-person(X23), -a_truth(statement_by(X23)), knight(X23)] [cut]
Reset 1
Start st4: [-person(X29), knight(X29), knave(X29)]
Extension ex1: -person(X29) -> [person(wife)] [cut]
Extension ex1: knight(X29) -> [-says(X47, Y47), -knight(X47), a_truth(Y47)] [cut]
Reset 1
Start st10: [-a_truth(statement_by(X62)), -person(X62), knight(X62)]
Extension ex5: -a_truth(statement_by(X62)) -> [a_truth(statement_by(husband)), -knight(wife)] [cut]
Start st15: [-knight(husband), knight(wife), -a_truth(statement_by(husband))]
Extension ex1: -knight(husband) -> [-person(X78), knight(X78), -a_truth(statement_by(X78))] [cut]
Extension ex1: knight(wife) -> [a_truth(Y82), -knight(X82), -says(X82, Y82)] [cut]
Reset 1
{'info': 'Theorem (SAT); (-person(husband), -a_truth(statement_by(husband)), knight(husband)); (a_truth(statement_by(husband)), -knight(wife)); (a_truth(statement_by(husband)), knight(wife)); (-knight(husband)); (person(husband))', 'steps': 10}
```

`compare\_strategies.py` can compare the result of running different strategies in a group of files.
The program takes a series of strategies and a directory or set of `.p` problems as an argument (all TPTP problems by default), and returns a CSV with solving information for each strategy/problem pair.

The program aggressively uses multiprocessing, and will try to ``fit'' each strategy/problem pair into a single processor core.
Additionally, it keep partial information of solved problems; runs of `compare\_strategies.py --name A` after the first one will retry from where the previous run ended, even if that run crashed or was killed for some reason.

The public `leancop_2` strategy implements the leanCoP 2.0 core features and
fixed schedule.  `compare_strategies.py` passes its per-problem timeout into
the strategy so that the ten schedule phases receive their published shares:

```
$ python compare_strategies.py --name LeanCoP2 --strategies leancop_2 $TPTP/Problems
```

```
$ python compare_strategies.py --name LongTest --strategies SATCoP SATResetCoP $TPTP/Problems/PUZ
Run: partial=partial/LongTest_PUZ  csv=results/LongTest/PUZ.csv  tasks=262  completed=0  remaining=262  workers=20  timeout=120s
Problem      Strategy     Result
--------------------------------
PUZ004-1.p   SATCoP       0 / 0ms
PUZ004-1.p   SATResetCoP  0 / 0ms
PUZ001-1.p   SATCoP       81 / 17ms
PUZ001-1.p   SATResetCoP  66 / 16ms
[...]

$ head results/LongTest/PUZ.csv | cut -d',' -f1-7 | column -t -s',' -R6,7
problem     strategy     status   result         solved   steps  time_ms
PUZ001+1.p  SATCoP       solved   Theorem (SAT)  True      2076      778
PUZ001+1.p  SATResetCoP  solved   Theorem (SAT)  True       989      821
PUZ001+2.p  SATCoP       timeout                 False    81106   120000
PUZ001+2.p  SATResetCoP  solved   Theorem (SAT)  True     35854    77695
PUZ001-1.p  SATCoP       solved   Theorem (SAT)  True        81       15
PUZ001-1.p  SATResetCoP  solved   Theorem (SAT)  True        66       15
PUZ001-2.p  SATCoP       timeout                 False   497938   120000
PUZ001-2.p  SATResetCoP  timeout                 False   513586   120000
PUZ001-3.p  SATCoP       timeout                 False   245563   120000
```
