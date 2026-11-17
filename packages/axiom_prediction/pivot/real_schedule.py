# Executes a schedule for real, one problem and prover at a time, within one wall-clock budget on one core: the plain
# prover runs for a 1 s probe and is paused (SIGSTOP); the network scores the problem if the schedule has learned
# stages; the stages run one after another with their slices; the plain prover is resumed (SIGCONT) for the rest.
# It checks the schedules that ladder_table.py rebuilds from separate stage runs.
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library_stages as stages_module
from library_stages import command, proved, syntax, clause_index, flatten, network_ranking, sine_selection
from axiom_prediction.prover_cores.collect_cores import die_with_parent, run_prover
from axiom_prediction.prover_cores.provers import Prover, modules
from axiom_prediction.search_workers import supervised_results

budget_s, probe_s, slice_s = 60, 1, 2
full_graph_limit = 20000
sine_settings = ((1, 1), (1, 2), (1, 3), (1, 0), (1.5, 0), (2, 0), (3, 0), (5, 0))
cutoffs = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
own_clauses = ('Prover9', 'iProver')

def write_filtered(source, prover, kept, target):
    pattern, keep = clause_index[syntax(prover)], set(kept)
    with open(target, 'w') as stream:
        for line in open(source):
            match = pattern.search(line)
            if match is None or int(match.group(1)) in keep:
                stream.write(line)

def start_plain(prover, path, log):
    return subprocess.Popen(
        command(prover, path, budget_s),
        stdout = log,
        stderr = subprocess.STDOUT,
        start_new_session = True,
        preexec_fn = die_with_parent,
    )

def finished(process, wait_s):
    try:
        process.wait(timeout = max(0, wait_s))
        return True
    except subprocess.TimeoutExpired:
        return False

def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

# One problem under one schedule: 'S' (SInE settings), 'L' (learned cut-offs) or 'SL' (both, in that order).
def run_schedule(key, root, scratch, checkpoint):
    import torch
    from axiom_prediction.tptp import load_tptp_problem

    path, prover, schedule = key.split('|')
    torch.set_num_threads(1)
    name = Path(path).name
    stem = f'{scratch}/{name}.{prover}.{schedule}'
    started = time.monotonic()
    problem = load_tptp_problem(path, tptp_root = root)
    conjectures = sorted(problem.conjecture_clause_ids)
    axioms = [i for i in range(len(problem.matrix.clauses)) if i not in set(conjectures)]
    flip = '-' not in name
    full = f'{stem}.{syntax(prover)}'
    modules[Prover.Prover9 if prover == 'Prover9' else Prover.E].write_input(problem.matrix, set(conjectures), flip, full, budget_s)
    original = f'{stem}.fof'
    flatten(path, root, original)
    load_s = time.monotonic() - started
    row = {'problem': key, 'axioms': len(axioms), 'load_s': round(load_s, 2), 'proved': False, 'by': '', 'score_s': 0.0, 'stages_run': 0}

    with open(f'{stem}.plain.log', 'w') as log:
        clock = time.monotonic()
        plain = start_plain(prover, full if prover in own_clauses else original, log)
        try:
            ended = finished(plain, probe_s)
            if ended and proved(prover, Path(f'{stem}.plain.log').read_text()):
                return {**row, 'proved': True, 'by': 'probe', 'seconds': round(time.monotonic() - clock, 3)}

            if not ended:
                os.killpg(plain.pid, signal.SIGSTOP)

            selections = []
            if 'S' in schedule:
                for tolerance, depth in sine_settings:
                    selected = sine_selection(problem.matrix, axioms, conjectures, tolerance, depth)
                    if 0 < len(selected) < len(axioms):
                        selections.append((f'esine-{tolerance}-{depth}', selected))

            # Problems over the size limit of the stage jobs are not scored, as there.
            if 'L' in schedule and len(axioms) <= full_graph_limit:
                scoring = time.monotonic()
                ranking, _ = network_ranking(problem.matrix, axioms, conjectures, checkpoint)
                row['score_s'] = round(time.monotonic() - scoring, 2)
                selections += [(f'learned-{k}', ranking[:k]) for k in cutoffs if len(axioms) > k]

            for label, selected in selections:
                window = min(slice_s, budget_s - (time.monotonic() - clock))
                if window <= 0.2:
                    break

                write_filtered(full, prover, selected + conjectures, f'{stem}.stage.p')
                output = run_prover(command(prover, f'{stem}.stage.p', window), window)
                row['stages_run'] += 1
                if proved(prover, output):
                    return {**row, 'proved': True, 'by': label, 'seconds': round(time.monotonic() - clock, 3)}

            if not ended:
                os.killpg(plain.pid, signal.SIGCONT)
                finished(plain, budget_s - (time.monotonic() - clock))

            stop(plain)
            log.flush()
            if proved(prover, Path(f'{stem}.plain.log').read_text()):
                return {**row, 'proved': True, 'by': 'resume', 'seconds': round(time.monotonic() - clock, 3)}

            return {**row, 'seconds': round(time.monotonic() - clock, 3)}
        finally:
            stop(plain)
            for suffix in ('plain.log', 'stage.p', 'fof', syntax(prover)):
                Path(f'{stem}.{suffix}').unlink(missing_ok = True)

def output_row(result):
    path, prover, schedule = result.pop('problem').split('|')
    row = {'problem': Path(path).name, 'prover': prover, 'schedule': schedule, **result}
    if 'outcome' in row:
        row['error'] = row.pop('outcome')

    return row

def main():
    parser = argparse.ArgumentParser(description = 'execute premise-selection schedules end to end')
    parser.add_argument('--problems', type = Path, required = True)
    parser.add_argument('--root', required = True)
    parser.add_argument('--checkpoint', required = True)
    parser.add_argument('--provers', nargs = '+', required = True)
    parser.add_argument('--schedules', nargs = '+', default = ['SL'], choices = ['S', 'L', 'SL'])
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--scratch', type = Path, required = True)
    parser.add_argument('--workers', type = int, default = 76)
    parser.add_argument('--count', type = int, default = 0)
    args = parser.parse_args()

    args.output.mkdir(parents = True, exist_ok = True)
    args.scratch.mkdir(parents = True, exist_ok = True)
    tempfile.tempdir = '/dev/shm'
    paths = args.problems.read_text().split()
    paths = paths[:args.count] if args.count else paths
    results = args.output / 'real.jsonl'
    done = set()
    if results.exists():
        done = {f"{r['problem']}|{r['prover']}|{r['schedule']}" for r in map(json.loads, results.open())}

    tasks = [
        (f'{path}|{prover}|{schedule}', args.root, str(args.scratch), args.checkpoint)
        for schedule in args.schedules
        for path in paths
        for prover in args.provers
        if f'{Path(path).name}|{prover}|{schedule}' not in done
    ]
    workers = min(args.workers, len(os.sched_getaffinity(0)))
    print(f'{len(tasks)} schedule runs to do, {len(done)} done, {workers} workers', flush = True)
    counts = {'finished': 0, 'proved': 0, 'errors': 0}
    rows = supervised_results(run_schedule, tasks, workers = workers, timeout = budget_s + 400)
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
