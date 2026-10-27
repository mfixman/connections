import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent = 2) + '\n')
    temporary.replace(path)

class Experiment:
    def __init__(self, args):
        self.args = args
        self.stopped = threading.Event()
        self.process_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.processes = []
        self.cpus = queue.Queue()
        for cpu in sorted(os.sched_getaffinity(0))[:args.workers]:
            self.cpus.put(cpu)

    def launch(self, path, stage, cpu, log, seconds, pause_clock = False):
        env = {k: v for k, v in os.environ.items() if not k.startswith('TOROK_')}
        env.update(TPTP = str(self.args.tptp), OMP_NUM_THREADS = '1',
                   OPENBLAS_NUM_THREADS = '1', MKL_NUM_THREADS = '1', PYTHONDONTWRITEBYTECODE = '1')
        mode = stage.get('mode', 'baseline')
        if pause_clock:
            env['TOROK_ACTIVE_LRS'] = '1'
        if stage.get('max_clauses', 512):
            env['TOROK_MAX_CLAUSES'] = str(stage.get('max_clauses', 512))
        command = ['stdbuf', '-oL', 'taskset', '-c', str(cpu), str(self.args.output / 'vampire'),
                   '--mode', stage.get('execution_mode', 'vampire'), '--time_limit', f'{max(.1, seconds):.1f}s',
                   '--memory_limit', str(stage.get('memory_mb', 2048)),
                   '--random_seed', str(stage.get('seed', 1)), '--proof', 'tptp',
                   '--saturation_algorithm', stage.get('algorithm', 'lrs'),
                   '--sine_selection', 'included' if 'included' in mode else 'off']
        if any(word in mode for word in ['filter', 'support', 'soft']):
            command += ['--torok_file', str(stage.get('model', self.args.output / 'torok.ts')), '--show_preprocessing', 'on']
        if 'filter' in mode:
            env['TOROK_FILTER'] = re.search(r'filter(\d+)', mode).group(1)
        if 'support' in mode:
            env['TOROK_SUPPORT'] = re.search(r'support(\d+)', mode).group(1)
            command += ['--sos', 'on' if 'soson' in mode else 'all']
        if 'soft' in mode:
            env['TOROK_SOFT'] = re.search(r'soft(\d+)', mode).group(1)
        if 'random' in mode:
            env['TOROK_RANDOM_SUPPORT'] = '1'
        if 'noavatar' in mode:
            command += ['--avatar', 'off']
        command += stage.get('extra', []) + [str(path)]
        with self.process_lock:
            if self.stopped.is_set():
                raise RuntimeError('Experiment deadline reached')
            process = subprocess.Popen(command, stdout = log, stderr = subprocess.STDOUT,
                                       env = env, start_new_session = True)
            self.processes.append(process)
        return process

    def wait(self, process, deadline):
        try:
            process.wait(timeout = max(0, deadline - time.monotonic()))
            return True
        except subprocess.TimeoutExpired:
            return False

    def stop(self, process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()

    def proved(self, path):
        return bool(re.search(r'SZS status (Theorem|Unsatisfiable|ContradictoryAxioms)\b',
                              path.read_text(errors = 'replace')))

    def vampire(self, path, config, cpu, folder):
        started = time.monotonic()
        deadline = started + self.args.seconds
        has_includes = bool(re.search(r'^\s*include\s*\(', path.read_text(), re.M))
        logs = []
        children = []
        stages = []
        alternatives = config.get('attempts', [])
        try:
            first_path = folder / 'ordinary.log'
            logs.append(first_path)
            with first_path.open('w') as output:
                first = self.launch(path, config.get('base', config), cpu, output,
                                    self.args.seconds, pause_clock = bool(alternatives))
                children.append(first)
                probe = config.get('probe_s', self.args.seconds)
                done = self.wait(first, min(deadline, started + probe))
                stages.append(dict(stage = 'ordinary', elapsed_s = time.monotonic() - started))
                if alternatives and not self.proved(first_path):
                    if not done:
                        os.killpg(first.pid, signal.SIGSTOP)
                    found = False
                    for index, stage in enumerate(alternatives):
                        if stage.get('requires_includes') and not has_includes:
                            continue
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or self.stopped.is_set():
                            break
                        log_path = folder / f'alternative-{index}.log'
                        logs.append(log_path)
                        allowance = stage['seconds'] if has_includes else stage.get('seconds_without_includes', stage['seconds'])
                        duration = min(allowance, remaining)
                        with log_path.open('w') as target:
                            child = self.launch(path, stage, cpu, target, duration)
                            children.append(child)
                            self.wait(child, min(deadline, time.monotonic() + duration))
                            self.stop(child)
                        stages.append(dict(stage = stage, elapsed_s = time.monotonic() - started))
                        if self.proved(log_path):
                            found = True
                            break
                    if not found and not done and not self.stopped.is_set():
                        os.killpg(first.pid, signal.SIGCONT)
                        self.wait(first, deadline)
        finally:
            for child in children:
                self.stop(child)
        texts = [p.read_text(errors = 'replace') for p in logs]
        return dict(proved = any(self.proved(p) for p in logs), elapsed_s = time.monotonic() - started,
                    proof_logs = [p.name for p in logs if self.proved(p)], stages = stages,
                    exit_codes = [p.returncode for p in children],
                    statuses = re.findall(r'SZS status (\w+)', '\n'.join(texts)),
                    errors = re.findall(r'User error: ([^\n]+)', '\n'.join(texts)))

    def prover9(self, path, config, cpu, folder):
        started = time.monotonic()
        command = ['taskset', '-c', str(cpu), sys.executable,
                   str(Path(__file__).with_name('prover9_baseline.py')), str(path),
                   '--prover9', str(self.args.prover9), '--tptp', str(self.args.tptp),
                   '--output', str(folder), '--seconds', str(self.args.seconds - .1),
                   '--memory-mb', str(config.get('memory_mb', 512))]
        with (folder / 'driver.log').open('w') as log:
            with self.process_lock:
                if self.stopped.is_set():
                    raise RuntimeError('Experiment deadline reached')
                child = subprocess.Popen(command, stdout = log, stderr = subprocess.STDOUT,
                                         start_new_session = True)
                self.processes.append(child)
            self.wait(child, started + self.args.seconds)
            self.stop(child)
        target = folder / 'native-result.json'
        result = json.loads(target.read_text()) if target.exists() else dict(proved = False, error = 'Timeout')
        log = folder / 'native.log'
        text = log.read_text(errors = 'replace') if log.exists() else ''
        complete = '============================== end of proof' in text.lower()
        result['proved'] = bool(result.get('proved') or complete)
        result.update(elapsed_s = time.monotonic() - started,
                      proof_logs = ['native.log'] if result['proved'] else [])
        return result

    def group(self, path, configs):
        cpu = self.cpus.get()
        try:
            offset = int(hashlib.md5(path.stem.encode()).hexdigest(), 16) % len(configs)
            order = configs[offset:] + configs[:offset]
            outcomes = {}
            for config in order:
                if self.stopped.is_set():
                    return
                folder = self.args.output / 'cases' / path.stem / config['name']
                folder.mkdir(parents = True, exist_ok = True)
                result = self.prover9(path, config, cpu, folder) if config.get('prover') == 'prover9' else self.vampire(path, config, cpu, folder)
                result.update(problem = path.name, config = config['name'], cpu = cpu)
                write_json(folder / 'result.json', result)
                outcomes[config['name']] = result
                row = [path.name, config['name'], result['proved'], round(1000 * result['elapsed_s']),
                       cpu, ','.join(result['proof_logs']), result.get('error', '')]
                with self.output_lock:
                    with (self.args.output / 'results.csv').open('a') as output:
                        csv.writer(output).writerow(row)
                    print(path.stem, config['name'], result['proved'], round(result['elapsed_s'], 2), flush = True)
            write_json(self.args.output / 'groups' / f'{path.stem}.json', outcomes)
        finally:
            self.cpus.put(cpu)

    def close(self):
        self.stopped.set()
        with self.process_lock:
            for process in self.processes:
                self.stop(process)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type = Path, required = True)
    parser.add_argument('--configs', type = Path, required = True)
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--seconds', type = float, default = 60)
    parser.add_argument('--workers', type = int, default = 20)
    parser.add_argument('--wall-limit-s', type = float, required = True)
    parser.add_argument('--vampire', type = Path, default = Path.home() / 'TP/vampire')
    parser.add_argument('--prover9', type = Path, default = Path.home() / 'TP/Prover9')
    parser.add_argument('--tptp', type = Path, default = Path.home() / 'TP/TPTP')
    args = parser.parse_args()
    started = time.monotonic()
    args.output = args.output.resolve()
    args.output.mkdir(parents = True, exist_ok = False)
    if not 1 <= args.workers <= min(20, len(os.sched_getaffinity(0))):
        raise ValueError('Worker count exceeds available CPUs or user limit')
    paths = list(map(Path, json.loads(args.sample.read_text())['problems']))
    configs = json.loads(args.configs.read_text())
    files = [args.vampire / 'build/vampire', args.vampire / 'Shell/Torok/torok.ts',
             args.prover9 / 'provers.src/prover9', args.prover9 / 'axiom_selection/build/native-search',
             Path(__file__), Path(__file__).with_name('prover9_baseline.py')]
    for config in configs:
        for stage in [config.get('base', config)] + config.get('attempts', []):
            if stage.get('model') and Path(stage['model']) not in files:
                files.append(Path(stage['model']))
    manifest = dict(problems = list(map(str, paths)), configs = configs, seconds = args.seconds,
                    workers = args.workers, cpus = sorted(os.sched_getaffinity(0))[:args.workers],
                    hashes = {str(p): digest(p) for p in files},
                    problem_hashes = {str(p): digest(p) for p in paths},
                    sample = json.loads(args.sample.read_text()), wall_limit_s = args.wall_limit_s,
                    protocol = 'Paired rotated arms, same pinned CPU per problem; one end-to-end wall budget per arm including cold scoring; line-buffered native logs; no free precomputed predictions.')
    write_json(args.output / 'manifest.json', manifest)
    shutil.copy2(files[0], args.output / 'vampire')
    shutil.copy2(files[1], args.output / 'torok.ts')
    for p in [Path(__file__), Path(__file__).with_name('prover9_baseline.py')]:
        shutil.copy2(p, args.output / p.name)
    (args.output / 'source.patch').write_bytes(subprocess.check_output(['git', '-C', str(args.vampire), 'diff', 'HEAD']))
    (args.output / 'groups').mkdir()
    with (args.output / 'results.csv').open('w') as output:
        csv.writer(output).writerow(['problem', 'config', 'proved', 'time_ms', 'cpu', 'proof_logs', 'error'])
    experiment = Experiment(args)
    executor = ThreadPoolExecutor(max_workers = args.workers)
    print(f'{len(paths)} problems, {len(configs)} configurations, {args.workers} workers', flush = True)
    futures = [executor.submit(experiment.group, path, configs) for path in paths]
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
        experiment.close()
        executor.shutdown(wait = True, cancel_futures = True)
        write_json(args.output / 'completion.json', dict(status = status,
                   elapsed_s = time.monotonic() - started,
                   completed_groups = len(list((args.output / 'groups').glob('*.json'))),
                   live_children = [p.pid for p in experiment.processes if p.poll() is None],
                   changed_sources = [str(p) for p in files if digest(p) != manifest['hashes'][str(p)]]))

if __name__ == '__main__':
    main()
