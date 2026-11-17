# Zero-shot premise selection on a TPTP-syntax library (MPTP2078 chainy): Connections loads and clausifies each problem,
# fixed scorers rank its axiom clauses, and each prover runs on the top-k clauses. One row per (problem, prover, arm, k).
# Phase 1 writes every problem once per input syntax with its rankings; phase 2 filters those files line by line.
# Mizar clauses share type guards, so the graph's complement edges grow quadratically (713k edges for 1.1k clauses):
# the network ranks the whole problem only up to full_graph_limit clauses ("learned"), and otherwise re-ranks the
# prefilter_size clauses with the highest symbol overlap ("rerank").
import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time

checkout = Path('/home/msf43/connections')
sys.path[:0] = [str(checkout / 'packages'), str(checkout / 'src')]

from axiom_prediction.prover_cores.collect_cores import run_prover
from axiom_prediction.prover_cores.provers import Prover, modules
from axiom_prediction.search_workers import supervised_results

satcop = '/home/msf43/connections/external/satcop/target/release/satcop'
spass = '/home/msf43/connections/external/SPASS/SPASS'
provers = ('Prover9', 'E', 'Vampire', 'SATCoP', 'iProver', 'SPASS', 'VampireSInE')
prefilter_size = 1024
clause_index = {
    'tptp': re.compile(r'^cnf\(c(\d+),'),
    'prover9': re.compile(r'# label\(weighted_input_(\d+)\)\.\s*$'),
}
loaded = {}

# External SInE settings as "tolerance:depth" (depth 0 = unlimited); the arm of a setting is esine-TOLERANCE-DEPTH.
def esine_arms(settings):
    return [f"esine-{setting.replace(':', '-')}" for setting in settings]

# (arm, k, limit in seconds), most important first: an unfinished job still has the comparison that matters.
def stage_plan(args):
    ranked = {'rerank': args.sizes, 'learned': [*args.sizes, 1024], 'shuffled': args.sizes, 'overlap': [*args.sizes, 1024]}
    fixed = {'plain': 60, 'sine': args.slice, 'bushy': 10, 'clausified': args.clausified_s, 'eclausified': 60}
    plan = []
    for arm in args.arms:
        if arm == 'esine':
            plan += [(setting, 0, args.slice) for setting in esine_arms(args.esine)]
            continue

        sizes = sorted(set(ranked[arm])) if arm in ranked else [0]
        plan += [(arm, k, fixed.get(arm, args.slice)) for k in sizes]

    return plan

def syntax(prover):
    return 'prover9' if prover == 'Prover9' else 'tptp'

def command(prover, path, seconds, own_sine = False):
    if prover == 'SATCoP':
        return [satcop, '--quiet', '--time', str(max(1, int(seconds))), str(path)]

    if prover == 'SPASS':
        return [spass, '-TPTP=1', f'-TimeLimit={max(1, int(seconds))}', '-PGiven=0', '-PProblem=0', str(path)]

    # VampireSInE is Vampire with its built-in SInE switched on in every run, so SInE works on whatever file it is
    # given: the whole problem in the plain run, the already filtered clauses in a stage run.
    if prover == 'VampireSInE':
        base = modules[Prover.Vampire].command(path, seconds)
        base[base.index('off')] = 'axioms'
        return base

    base = modules[Prover(prover)].command(path, seconds)
    if own_sine and prover == 'E':
        base.insert(2, '--sine=Auto')
    elif own_sine:
        base[base.index('off')] = 'axioms'

    return base

def proved(prover, output):
    if prover == 'SATCoP':
        return bool(re.search(r'SZS status (Unsatisfiable|Theorem)', output))

    if prover == 'SPASS':
        return 'SPASS beiseite: Proof found.' in output

    if prover == 'VampireSInE':
        return modules[Prover.Vampire].proved(output)

    return modules[Prover(prover)].proved(output)

# Inlines whole-file includes (axiom files include each other); a selective include stays for the prover to resolve.
# MPTP2078 uses bare numerals as constants, which E and Vampire read as integers and reject in first-order input
# (217 of the 300 pilot problems failed at once); rename_numerals turns them into ordinary constants.
def flatten(path, root, target, rename_numerals = False):
    include = re.compile(r"^include\('([^']+)'\)\.[ \t]*$", re.M)
    text = Path(path).read_text()
    for _ in range(6):
        text, replaced = include.subn(lambda match: (Path(root) / match.group(1)).read_text(), text)
        if not replaced:
            break

    if rename_numerals:
        text = re.sub(r"(?<![A-Za-z0-9_$'.])(\d+)(?![A-Za-z0-9_'.])", r"numeral_\1", text)

    Path(target).write_text(text)

def scripted(checkpoint):
    import torch
    if checkpoint not in loaded:
        loaded[checkpoint] = torch.jit.load(checkpoint, map_location = 'cpu').eval()

    return loaded[checkpoint]

def rank(axioms, scores):
    return [index for _, index in sorted(zip(map(float, scores), axioms), key = lambda pair: (-pair[0], pair[1]))]

# SInE (Hoder & Voronkov 2011) on clauses, outside any prover: a symbol triggers a clause if it occurs in it and is at
# most `tolerance` times as frequent as the clause's rarest symbol; starting from the conjecture's symbols, triggered
# clauses are added for `depth` rounds (0 = until nothing changes). Equality is not counted as a symbol.
def sine_selection(matrix, axioms, conjectures, tolerance, depth):
    from collections import Counter, defaultdict
    from axiom_prediction.sine import matrix_symbols

    symbols = [{s for s in clause if s[0] not in ('=', 'equal___')} for clause in matrix_symbols(matrix)]
    frequency = Counter(s for index in axioms for s in symbols[index])
    triggers = defaultdict(list)
    for index in axioms:
        if symbols[index]:
            rarest = min(frequency[s] for s in symbols[index])
            for s in symbols[index]:
                if frequency[s] <= tolerance * rarest:
                    triggers[s].append(index)

    frontier = set().union(*[symbols[index] for index in conjectures])
    seen, selected, rounds = set(frontier), set(), 0
    while frontier and (depth == 0 or rounds < depth):
        reached = set()
        for s in frontier:
            for index in triggers.get(s, ()):
                if index not in selected:
                    selected.add(index)
                    reached |= symbols[index] - seen

        seen |= reached
        frontier, rounds = reached, rounds + 1

    return sorted(selected)

# Fraction of a clause's symbols that occur in the conjecture clauses.
def overlap_scores(matrix, axioms, conjectures):
    from axiom_prediction.sine import matrix_symbols

    symbols = matrix_symbols(matrix)
    goal = set().union(*[symbols[index] for index in conjectures])
    return [len(symbols[index] & goal) / max(len(symbols[index]), 1) for index in axioms]

# Ranks the given axiom clauses with the network, on a graph of those clauses and the conjectures only.
def network_ranking(matrix, axioms, conjectures, checkpoint):
    import torch
    from axiom_prediction.graph import build_axiom_graph
    from axiom_prediction.model import scripted_logits
    from connections.syntax.matrix import Matrix

    keep = sorted(set(axioms) | set(conjectures))
    position = {index: place for place, index in enumerate(keep)}
    part = Matrix(tuple(matrix.clauses[index] for index in keep), matrix.logic, matrix.domain, matrix.source_has_conjecture)
    local_axioms = [position[index] for index in axioms]
    graph = build_axiom_graph(part, axiom_clause_ids = local_axioms, conjecture_clause_ids = [position[index] for index in conjectures])
    with torch.inference_mode():
        scores = scripted_logits(scripted(checkpoint), graph).tolist()

    return rank(axioms, scores), len(graph.graph.edges['complement'])

# Runs in a supervised worker: one load and at most two scorings per problem.
def prepare(path, root, bushy, scratch, checkpoint, full_graph_limit, arms, rename_numerals, esine = (), clause_form = False):
    import torch
    from axiom_prediction.sine import score_matrix
    from axiom_prediction.tptp import load_tptp_problem

    torch.set_num_threads(1)
    name = Path(path).name
    stem = f'{scratch}/{name}'
    timer = time.monotonic()
    flatten(path, root, f'{stem}.fof', rename_numerals)
    problem = load_tptp_problem(path, tptp_root = root)
    conjectures = sorted(problem.conjecture_clause_ids)
    marked = set(conjectures)
    axioms = [i for i in range(len(problem.matrix.clauses)) if i not in marked]
    # Connections stores a first-order problem as the DNF of the theorem, so every literal is flipped for these provers;
    # TPTP's clause-form problems (names with "-") are kept as they are.
    flip = '-' not in name and not clause_form
    modules[Prover.E].write_input(problem.matrix, marked, flip, f'{stem}.tptp', 60)
    modules[Prover.Prover9].write_input(problem.matrix, marked, flip, f'{stem}.prover9', 60)
    bushy_axioms = None
    if bushy:
        flatten(f'{bushy}/{name}', root, f'{stem}.bushy.fof', rename_numerals)
        small = load_tptp_problem(f'{bushy}/{name}', tptp_root = root)
        modules[Prover.Prover9].write_input(small.matrix, set(small.conjecture_clause_ids), flip, f'{stem}.bushy.prover9', 10)
        bushy_axioms = len(small.matrix.clauses) - len(small.conjecture_clause_ids)

    timings = {'load': round(time.monotonic() - timer, 2)}
    # A second clause form for the provers that cannot read first-order input: E's clausifier on the original file.
    e_clauses = None
    if 'eclausified' in arms:
        clausifier = [str(checkout / 'external/eprover/bin/eprover'), '--cnf', '--tstp-format', '-s', f'{stem}.fof']
        kept = [line for line in run_prover(clausifier, 60).splitlines() if line.startswith('cnf(')]
        if kept:
            Path(f'{stem}.ecnf').write_text('\n'.join(kept) + '\n')
            other = load_tptp_problem(f'{stem}.ecnf', tptp_root = root)
            modules[Prover.Prover9].write_input(other.matrix, set(other.conjecture_clause_ids), False, f'{stem}.ecnf.prover9', 60)
            e_clauses = len(other.matrix.clauses)


    selected = score_matrix(problem.matrix, axioms, conjectures)
    rankings = {'sine': [index for index, score in zip(axioms, selected) if score > 0]}
    shuffled = list(axioms)
    random.Random(name).shuffle(shuffled)
    rankings['shuffled'] = shuffled[:prefilter_size]
    rankings['overlap'] = rank(axioms, overlap_scores(problem.matrix, axioms, conjectures))[:prefilter_size]
    for setting, arm in zip(esine, esine_arms(esine)):
        tolerance, depth = setting.split(':')
        rankings[arm] = sine_selection(problem.matrix, axioms, conjectures, float(tolerance), int(depth))

    edges = {}
    if 'learned' in arms and len(axioms) <= full_graph_limit:
        timer = time.monotonic()
        rankings['learned'], edges['learned'] = network_ranking(problem.matrix, axioms, conjectures, checkpoint)
        rankings['learned'] = rankings['learned'][:prefilter_size]
        timings['learned'] = round(time.monotonic() - timer, 2)

    if 'rerank' in arms and len(axioms) > prefilter_size:
        timer = time.monotonic()
        rankings['rerank'], edges['rerank'] = network_ranking(problem.matrix, rankings['overlap'], conjectures, checkpoint)
        timings['rerank'] = round(time.monotonic() - timer, 2)

    return {
        'problem': name, 'axioms': len(axioms), 'conjectures': conjectures, 'bushy_axioms': bushy_axioms, 'e_clauses': e_clauses,
        'rankings': rankings, 'timings': timings, 'edges': edges,
    }

def run_task(key, source, kept, seconds):
    name, prover, arm, k = key.split('|')
    with tempfile.NamedTemporaryFile('w', suffix = '.p') as filtered:
        path = source
        if kept is not None:
            pattern, keep = clause_index[syntax(prover)], set(kept)
            for line in open(source):
                match = pattern.search(line)
                if match is None or int(match.group(1)) in keep:
                    filtered.write(line)

            filtered.flush()
            path = filtered.name

        own_sine = arm == 'sine' and kept is None
        started = time.monotonic()
        output = run_prover(command(prover, path, seconds, own_sine), seconds + 1)
        run_s = time.monotonic() - started

    # SPASS prints no SZS status; its closing line ("Proof found", "Completion found", "Ran out of time") stands in.
    found = re.findall(r'SZS status (\w+)|SPASS beiseite: ([^.\n]+)', output)
    status = ','.join(sorted({szs or closing for szs, closing in found}))
    return {'problem': key, 'proved': proved(prover, output), 'run_s': round(run_s, 3), 'kept': len(kept or ()), 'status': status}

def task_input(prepared, name, prover, arm, k, scratch, bushy):
    stem = f'{scratch}/{name}'
    clausified = f'{stem}.{syntax(prover)}'
    ranking = prepared['rankings']
    # Prover9 does not read TPTP and this iProver build has no clausifier for first-order input: both always get the
    # Connections clause set, so their plain run is the clausified one.
    own_clauses = prover in ('Prover9', 'iProver')
    if arm == 'plain':
        return (clausified if own_clauses else f'{stem}.fof'), None

    if arm == 'bushy' and not bushy:
        return None, None

    if arm == 'bushy':
        return (f'{stem}.bushy.prover9' if prover == 'Prover9' else f'{stem}.bushy.fof'), None

    if arm == 'clausified':
        return (None if own_clauses else clausified), None

    if arm == 'eclausified':
        usable = own_clauses and prepared.get('e_clauses')
        return ((f'{stem}.ecnf.prover9' if prover == 'Prover9' else f'{stem}.ecnf') if usable else None), None

    if arm == 'sine' and prover in ('E', 'Vampire'):
        return f'{stem}.fof', None

    if arm == 'sine' or arm.startswith('esine-'):
        usable = 0 < len(ranking[arm]) < prepared['axioms']
        return (clausified if usable else None), ranking[arm] + prepared['conjectures']

    if prepared['axioms'] <= k or arm not in ranking:
        return None, None

    return clausified, ranking[arm][:k] + prepared['conjectures']

def output_row(result):
    name, prover, arm, k = result.pop('problem').split('|')
    row = {'problem': name, 'prover': prover, 'arm': arm, 'k': int(k), **result}
    row['seconds'] = round(row['seconds'], 2)
    if 'outcome' in row:
        row['error'] = row.pop('outcome')

    return row

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'zero-shot premise-selection stages on a TPTP-syntax library')
    parser.add_argument('--problems', type = Path, required = True, help = 'file with one problem path per line')
    parser.add_argument('--root', required = True, help = 'directory that include() paths are relative to')
    parser.add_argument('--bushy', help = 'directory with the same problems under minimised premises')
    parser.add_argument('--arms', nargs = '+', default = ['plain', 'rerank', 'learned', 'shuffled', 'sine', 'overlap', 'bushy', 'clausified'])
    parser.add_argument('--sizes', type = int, nargs = '+', default = [16, 64, 256], help = 'top-k cut-offs of the ranked arms')
    parser.add_argument('--slice', type = float, default = 8, help = 'seconds per ranked stage')
    parser.add_argument('--full-graph-limit', type = int, default = 2048, help = 'largest problem the network ranks whole')
    parser.add_argument('--prepare-timeout', type = float, default = 600, help = 'seconds to load and score one problem')
    parser.add_argument('--clausified-s', type = float, default = 10, help = 'limit of the run on the unfiltered Connections clause set')
    parser.add_argument('--rename-numerals', action = 'store_true', help = 'bare numerals become constants in the original files')
    parser.add_argument('--clause-form', action = 'store_true', help = 'the problem files hold clauses whatever their names say')
    parser.add_argument('--esine', nargs = '+', default = ['1:1', '1:2', '1:3', '1:0', '1.5:0', '2:0', '3:0', '5:0'],
        help = 'external SInE settings TOLERANCE:DEPTH for the esine arm (depth 0 = unlimited)')
    parser.add_argument('--checkpoint', required = True, help = 'TorchScript scorer (a chunked export bounds memory)')
    parser.add_argument('--provers', nargs = '+', default = list(provers[:4]), choices = provers)
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--scratch', type = Path, required = True)
    parser.add_argument('--workers', type = int, default = 76)
    parser.add_argument('--count', type = int, default = 0)
    parser.add_argument('--limit-scale', type = float, default = 1, help = 'multiplies every time limit, for a check')
    return parser.parse_args()

def main():
    args = parse_arguments()
    args.output.mkdir(parents = True, exist_ok = True)
    args.scratch.mkdir(parents = True, exist_ok = True)
    tempfile.tempdir = '/dev/shm'
    paths = args.problems.read_text().split()
    paths = paths[:args.count] if args.count else paths
    workers = min(args.workers, len(os.sched_getaffinity(0)))

    prepared = {}
    settings = (args.root, args.bushy, str(args.scratch), args.checkpoint, args.full_graph_limit, args.arms, args.rename_numerals)
    settings += (tuple(args.esine) if 'esine' in args.arms else (), args.clause_form)
    prepare_arguments = [(path, *settings) for path in paths]
    for result in supervised_results(prepare, prepare_arguments, workers = workers, timeout = args.prepare_timeout):
        prepared[Path(result['problem']).name] = result

    ready = {name: p for name, p in prepared.items() if 'rankings' in p}
    print(f'prepared {len(ready)} of {len(paths)} problems', flush = True)
    (args.output / 'prepared.json').write_text(json.dumps(prepared))

    results = args.output / 'results.jsonl'
    done = set()
    if results.exists():
        done = {f"{r['problem']}|{r['prover']}|{r['arm']}|{r['k']}" for r in map(json.loads, results.open())}

    arguments = []
    for arm, k, limit in stage_plan(args):
        for name in sorted(ready):
            for prover in args.provers:
                key = f'{name}|{prover}|{arm}|{k}'
                source, kept = task_input(ready[name], name, prover, arm, k, args.scratch, args.bushy)
                if source is not None and key not in done:
                    arguments.append((key, source, kept, limit * args.limit_scale))

    print(f'{len(arguments)} tasks to do, {len(done)} done, {workers} workers', flush = True)
    counts = {'finished': 0, 'proved': 0, 'errors': 0}
    rows = supervised_results(run_task, arguments, workers = workers, timeout = 90)
    try:
        with results.open('a') as stream:
            for row in map(output_row, rows):
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                counts['finished'] += 1
                counts['proved'] += bool(row.get('proved'))
                counts['errors'] += 'error' in row
                if counts['finished'] % 500 == 0:
                    print(counts, flush = True)
    finally:
        rows.close()

    print(f'finished: {counts}', flush = True)

if __name__ == '__main__':
    main()
