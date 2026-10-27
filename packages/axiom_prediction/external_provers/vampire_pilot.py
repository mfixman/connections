import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent = 2) + '\n')
    temporary.replace(path)

def run_group(problem, args, cpu, runner, modes, stopped, output_lock):
    path = Path(problem)
    settings = SimpleNamespace(output = args.output, tptp = args.tptp, seconds = 60,
                               probe = 40, fraction = 1 / 3, max_clauses = 256,
                               seed = 1, algorithm = 'lrs', extra = [])
    offset = int(runner.digest(path)[:8], 16) % len(modes)
    order = modes[offset:] + modes[:offset]
    outcomes = {}
    for mode in order:
        if stopped.is_set():
            return

        result = runner.run_case(path, mode, settings, cpu)
        outcomes[mode] = result
        row = dict(problem = path.name, mode = mode, proved = result['solved'],
                   time_ms = round(1000 * result['elapsed']), cpu = cpu,
                   proof_logs = ','.join(result['proof_logs']))
        with output_lock:
            with (args.output / 'results.csv').open('a') as output:
                csv.DictWriter(output, fieldnames = row).writerow(row)
            print(path.stem, mode, result['solved'], round(result['elapsed'], 2), flush = True)

    write_json(args.output / 'groups' / f'{path.stem}.json', outcomes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--vampire', type = Path, default = Path.home() / 'TP/vampire')
    parser.add_argument('--tptp', type = Path, default = Path.home() / 'TP/TPTP')
    parser.add_argument('--workers', type = int, default = 18)
    parser.add_argument('--wall-limit-s', type = float, default = 990)
    args = parser.parse_args()
    started = time.monotonic()
    args.output = args.output.resolve()
    source = args.vampire / 'Shell/Torok/experiments/compare.py'
    spec = importlib.util.spec_from_file_location('vampire_compare', source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cpus = sorted(os.sched_getaffinity(0))[:args.workers]
    if not 1 <= args.workers <= min(18, len(cpus)):
        raise ValueError('Use at most 18 available workers')

    problems = json.loads((args.output / 'sample.json').read_text())['problems']
    modes = ['baseline', 'resume-support1:20', 'resume-random-support1:20',
             'resume-support1-soson:20', 'resume-random-support1-soson:20']
    files = [args.vampire / 'build/vampire', args.vampire / 'Shell/Torok/torok.ts', source]
    files += list((args.vampire / 'Shell/Torok').glob('*.cpp'))
    files += list((args.vampire / 'Shell/Torok').glob('*.hpp'))
    files += [args.vampire / 'Saturation/SaturationAlgorithm.cpp', Path(__file__).resolve()]
    manifest = dict(problems = problems, modes = modes, seconds = 60, probe_s = 40,
                    guided_allowance_s = 20, max_clauses = 256, memory_mb = 2048,
                    native_seed = 1, random_ranking_seed = 1, algorithm = 'lrs',
                    cpus = cpus, wall_limit_s = args.wall_limit_s,
                    hashes = {str(p): runner.digest(p) for p in files},
                    problem_hashes = {p: runner.digest(Path(p)) for p in problems},
                    revision = subprocess.check_output(['git', '-C', str(args.vampire),
                                                        'rev-parse', 'HEAD'], text = True).strip(),
                    protocol = 'Five paired rotated arms on the same CPU; cold native scoring inside the final 20s; one 60s deadline; paused baseline resumes after early guided failure. No SInE; default AVATAR retained.')
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Use a new output directory for each pilot')

    write_json(args.output / 'manifest.json', manifest)
    shutil.copy2(files[0], args.output / 'vampire')
    shutil.copy2(files[1], args.output / 'torok.ts')
    shutil.copy2(source, args.output / 'compare.py')
    shutil.copy2(__file__, args.output / 'vampire_pilot.py')
    (args.output / 'source.patch').write_bytes(subprocess.check_output(
        ['git', '-C', str(args.vampire), 'diff', 'HEAD']))
    (args.output / 'groups').mkdir()
    with (args.output / 'results.csv').open('w') as output:
        csv.writer(output).writerow(['problem', 'mode', 'proved', 'time_ms', 'cpu', 'proof_logs'])

    stopped = threading.Event()
    process_lock = threading.Lock()
    output_lock = threading.Lock()
    processes = []
    original_launch = runner.launch
    def launch(*positional, **keywords):
        with process_lock:
            if stopped.is_set():
                raise RuntimeError('Pilot deadline reached')

            process = original_launch(*positional, **keywords)
            processes.append(process)
            return process

    runner.launch = launch
    executor = ThreadPoolExecutor(max_workers = len(cpus))
    def bucket(cpu, paths):
        for problem in paths:
            if stopped.is_set():
                return

            run_group(problem, args, cpu, runner, modes, stopped, output_lock)

    print(f'Starting {len(problems)} problems, {len(modes)} arms, {len(cpus)} workers', flush = True)
    futures = [executor.submit(bucket, cpu, problems[index::len(cpus)])
               for index, cpu in enumerate(cpus)]
    status = 'complete'
    try:
        for future in as_completed(futures, timeout = args.wall_limit_s):
            future.result()
    except TimeoutError:
        status = 'wall_limit'
    except BaseException:
        status = 'error'
        raise
    finally:
        stopped.set()
        with process_lock:
            for process in processes:
                runner.stop(process)
        executor.shutdown(wait = True, cancel_futures = True)
        changed = [str(p) for p in files if runner.digest(p) != manifest['hashes'][str(p)]]
        write_json(args.output / 'completion.json', dict(status = status,
                   elapsed_s = time.monotonic() - started,
                   completed_groups = len(list((args.output / 'groups').glob('*.json'))),
                   live_children = [p.pid for p in processes if p.poll() is None],
                   changed_sources = changed))

if __name__ == '__main__':
    main()
