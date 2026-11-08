# Reruns the problems a Prover9 strategy gained over the baseline and records which stage proved each one.
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import tempfile

import paths
import results

search_labels = [('filtered_search', 'top-16 filter'), ('guided_search', 'top-1 support'), ('baseline_search', 'plain Prover9')]
predictor = None

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'stage attribution of the problems gained over the Prover9 baseline')
    parser.add_argument('--run', type = Path, default = paths.goal / 'prover9/part9-test', help = 'result directory with results-MODE.csv files')
    parser.add_argument('--manifest', type = Path, default = paths.goal / 'manifests/part9-cnf-unsat.json')
    parser.add_argument('--model', default = str(paths.big_train / 'Sharbaraz1/model/epoch-0100.ts'))
    parser.add_argument('--modes', nargs = '+', default = ['ResumeFilter16Support1', 'ShuffledResumeFilter16Support1'])
    parser.add_argument('--tptp', type = Path, default = paths.tptp)
    parser.add_argument('--prover9', type = Path, default = paths.prover9_selection, help = "Prover9's axiom_selection directory")
    parser.add_argument('--scratch', type = Path, default = paths.goal / 'scratch')
    parser.add_argument('--workers', type = int, default = 32)
    parser.add_argument('--seconds', type = float, default = 59.95)
    parser.add_argument('--output', type = Path, help = 'defaults to gain-stages.json inside --run')
    return parser.parse_args()

def stage(task):
    path, mode, settings = task
    paths.add_prover9(settings['prover9'])
    from premise_run import Mode, solve
    from predict import Predictor
    global predictor
    # Torch allows one thread setup per process, so each pool process keeps its predictor.
    if predictor is None:
        predictor = Predictor(settings['model'])

    with tempfile.TemporaryDirectory(dir = settings['scratch']) as directory:
        result = solve(Path(path), Path(settings['tptp']), predictor, Mode(mode), settings['seconds'], directory)

    found = lambda key: bool((result.get(key) or {}).get('proved'))
    return path, mode, next((label for key, label in search_labels if found(key)), 'not proved')

def gained(run, modes):
    base = results.proved(run / 'results-Baseline.csv')
    return [(mode, sorted(results.proved(run / f'results-{mode}.csv') - base)) for mode in modes]

def main():
    args = parse_arguments()
    settings = {k: str(getattr(args, k)) for k in ('prover9', 'model', 'scratch', 'tptp')} | {'seconds': args.seconds}
    paths_by_name = {Path(p).name: p for p in json.load(open(args.manifest))['paths']}
    tasks = [(paths_by_name[p], mode, settings) for mode, problems in gained(args.run, args.modes) for p in problems]
    with ProcessPoolExecutor(args.workers, mp_context = multiprocessing.get_context('spawn')) as pool:
        stages = list(pool.map(stage, tasks))

    for mode in args.modes:
        counts = {}
        for _, m, label in stages:
            if m == mode:
                counts[label] = counts.get(label, 0) + 1

        print(mode, counts, flush = True)

    json.dump(stages, open(args.output or args.run / 'gain-stages.json', 'w'), indent = 1)

if __name__ == '__main__':
    main()
