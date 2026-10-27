import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    modes = manifest['modes']
    groups = {p.stem: json.loads(p.read_text()) for p in (directory / 'groups').glob('*.json')}
    expected = {Path(p).stem for p in manifest['problems']}
    if not set(groups) <= expected:
        raise ValueError('Unexpected problem in completed groups')

    counts = {}
    mismatches = []
    incomplete_proofs = []
    attribution = {}
    for mode in modes:
        values = [g[mode] for g in groups.values()]
        stages = Counter()
        skipped = []
        selected = 0
        attempts = 0
        for problem, group in groups.items():
            result = group[mode]
            folder = directory / problem / mode
            logs = {p.name: p.read_text(errors = 'replace') for p in folder.glob('*.log')}
            proved = [name for name, text in logs.items() if re.search(
                r'SZS status (Theorem|Unsatisfiable|ContradictoryAxioms)\b', text)]
            if bool(proved) != result['solved'] or sorted(proved) != sorted(result['proof_logs']):
                mismatches.append([problem, mode])
            for name in proved:
                if not ('SZS output start Proof' in logs[name] and 'SZS output end Proof' in logs[name]):
                    incomplete_proofs.append([problem, mode, name])
            if result['cpu'] not in manifest['cpus']:
                raise ValueError('Unexpected CPU in result')

            text = logs.get('guided.log', '')
            attempts += 'guided.log' in logs
            selected += bool(re.search(r'Torok ranking: selected \d+ of \d+ axioms', text))
            if 'Torok clause budget exceeded' in text:
                skipped.append(problem)
            if result['solved']:
                stage = 'guided' if 'guided.log' in proved else 'ordinary'
                stages[stage] += 1
                attribution.setdefault(mode, {})[problem] = stage

        counts[mode] = dict(proofs = sum(v['solved'] for v in values),
                            par2_s = statistics.mean(v['elapsed'] if v['solved'] else 120
                                                    for v in values) if values else None,
                            stages = dict(stages), guided_attempts = attempts,
                            logged_support_selections = selected,
                            clause_cap_skips = skipped,
                            maximum_elapsed_s = max((v['elapsed'] for v in values), default = 0),
                            termination_reasons = dict(Counter(r for v in values for r in v['reasons'])))

    comparisons = {}
    references = {'baseline': modes[1:],
                  'resume-random-support1:20': ['resume-support1:20'],
                  'resume-random-support1-soson:20': ['resume-support1-soson:20']}
    for reference, alternatives in references.items():
        base = {p for p, g in groups.items() if g[reference]['solved']}
        for mode in alternatives:
            solved = {p for p, g in groups.items() if g[mode]['solved']}
            gains = sorted(solved - base)
            comparisons[f'{mode} vs {reference}'] = dict(
                gains = gains, losses = sorted(base - solved), net = len(solved) - len(base),
                gain_stages = {p: attribution[mode][p] for p in gains})

    return dict(expected_groups = len(expected), completed_groups = len(groups),
                missing_groups = sorted(expected - set(groups)), modes = counts,
                comparisons = comparisons, attribution = attribution,
                proof_log_mismatches = mismatches, incomplete_proofs = incomplete_proofs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type = Path)
    parser.add_argument('--save', action = 'store_true')
    args = parser.parse_args()
    summary = summarize(args.directory)
    print(f"Completed paired groups: {summary['completed_groups']}/{summary['expected_groups']}")
    for mode, result in summary['modes'].items():
        print(mode, result['proofs'], 'proofs;', result['stages'],
              'PAR2', round(result['par2_s'], 3) if result['par2_s'] else None)
    for label, result in summary['comparisons'].items():
        print(label, result)
    print('Proof-log mismatches:', summary['proof_log_mismatches'])
    if args.save:
        (args.directory / 'summary.json').write_text(json.dumps(summary, indent = 2) + '\n')

if __name__ == '__main__':
    main()
