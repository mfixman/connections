# Readers for the goal's result CSVs (SATCoP and Prover9 formats), with the paired statistics the tables use.
import csv
import math
from pathlib import Path
import re

def rows(path):
    with open(path, newline = '') as handle:
        return list(csv.DictReader(handle))

def is_proved(row):
    return row['proved'].lower() == 'true'

def has_error(row):
    return row.get('error', '') not in ('', 'false')

def proved(path, key = lambda row: row['problem']):
    return {key(row) for row in rows(path) if is_proved(row)}

def times(path):
    return {row['problem']: float(row['seconds']) for row in rows(path) if is_proved(row)}

def family(problem):
    # TPTP family of a problem path or file name, e.g. SYN003 for SYN003-1.p.
    return re.match(r'[A-Z]{3}\d{3}', Path(problem).name).group(0)

def families(problems):
    return set(map(family, problems))

def pct(count, base):
    return f'{100 * (count - base) / base:+.2f}%'

def mcnemar(gains, losses):
    # Exact two-sided McNemar test on the discordant pairs.
    n = gains + losses
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
