import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

def stage_name(config, log):
    if log.startswith('alternative-'):
        index = int(re.search(r'\d+', log).group())
        stage = config['attempts'][index]
    else:
        stage = config.get('base', config)
    if stage.get('execution_mode') == 'casc':
        return 'CASC'
    if '--age_weight_ratio' in stage.get('extra', []) and '--selection' in stage.get('extra', []):
        return 'lookahead_weight10'
    return stage.get('mode', stage.get('name', 'baseline'))

def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    configs = {c['name']: c for c in manifest['configs']}
    groups = {p.stem: json.loads(p.read_text()) for p in (directory / 'groups').glob('*.json')}
    expected = {Path(p).stem for p in manifest['problems']}
    summary = dict(expected_groups = len(expected), completed_groups = len(groups),
                   missing_groups = sorted(expected - set(groups)), configs = {}, comparisons = {},
                   attribution = {}, proof_mismatches = [], incomplete_proofs = [])
    for name, config in configs.items():
        values = [g[name] for g in groups.values()]
        stages = Counter()
        selection_proofs = []
        errors = Counter()
        for problem, group in groups.items():
            result = group[name]
            folder = directory / 'cases' / problem / name
            logs = {p.name: p.read_text(errors = 'replace') for p in folder.glob('*.log')}
            is_prover9 = config.get('prover') == 'prover9'
            observed = []
            for log, text in logs.items():
                if is_prover9:
                    proved = bool(re.search(r'={5,}\s*end of proof', text, re.I))
                    complete = proved and bool(re.search(r'={5,}\s*PROOF\s*={5,}', text))
                else:
                    proved = bool(re.search(r'SZS status (Theorem|Unsatisfiable|ContradictoryAxioms)\b', text))
                    complete = 'SZS output start Proof' in text and 'SZS output end Proof' in text
                if proved:
                    observed.append(log)
                    if not complete:
                        summary['incomplete_proofs'].append([problem, name, log])
            if bool(observed) != result['proved'] or sorted(observed) != sorted(result['proof_logs']):
                summary['proof_mismatches'].append([problem, name])
            errors.update(result.get('errors', []))
            if result.get('error'):
                errors[result['error']] += 1
            if result['proved']:
                log = result['proof_logs'][0]
                stage = 'Prover9' if is_prover9 else stage_name(config, log)
                stages[stage] += 1
                summary['attribution'].setdefault(name, {})[problem] = stage
                match = re.search(r'Torok ranking: selected (\d+) of (\d+) axioms', logs[log])
                if match:
                    selected, axioms = map(int, match.groups())
                    selection_proofs.append(dict(problem = problem, stage = stage,
                                                 selected = selected, axioms = axioms))
        summary['configs'][name] = dict(proofs = sum(v['proved'] for v in values),
            par2_s = statistics.mean(v['elapsed_s'] if v['proved'] else 2 * manifest['seconds']
                                    for v in values) if values else None,
            maximum_elapsed_s = max((v['elapsed_s'] for v in values), default = 0),
            stages = dict(stages), errors = dict(errors), selection_proofs = selection_proofs)
    for reference in [n for n in ['LRS', 'Prover9', 'Prover9_2GB', 'ModelFree', 'Random'] if n in configs]:
        base = {p for p, g in groups.items() if g[reference]['proved']}
        for name in configs:
            if name == reference:
                continue
            solved = {p for p, g in groups.items() if g[name]['proved']}
            gains, losses = sorted(solved - base), sorted(base - solved)
            discordant = len(gains) + len(losses)
            p_value = min(1, 2 * sum(math.comb(discordant, k) for k in range(min(len(gains), len(losses)) + 1)) / 2 ** discordant) if discordant else 1
            summary['comparisons'][f'{name} vs {reference}'] = dict(gains = gains, losses = losses,
                net = len(solved) - len(base), relative_percent = 100 * (len(solved) - len(base)) / len(base) if base else None,
                discordance_p = p_value,
                gain_stages = {p: summary['attribution'][name][p] for p in gains})
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type = Path)
    parser.add_argument('--save', action = 'store_true')
    args = parser.parse_args()
    summary = summarize(args.directory)
    print(f"Paired groups: {summary['completed_groups']}/{summary['expected_groups']}")
    for name, value in sorted(summary['configs'].items(), key = lambda item: (-item[1]['proofs'], item[1]['par2_s'] or 0)):
        print(name, value['proofs'], 'PAR2', round(value['par2_s'], 3) if value['par2_s'] else None,
              'stages', value['stages'])
    for label, value in summary['comparisons'].items():
        if label.endswith('vs LRS'):
            print(label, value)
    print('Proof mismatches', summary['proof_mismatches'], 'incomplete proofs', summary['incomplete_proofs'])
    if args.save:
        write = args.directory / 'summary.json'
        write.write_text(json.dumps(summary, indent = 2) + '\n')

if __name__ == '__main__':
    main()
