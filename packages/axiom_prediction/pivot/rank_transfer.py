# Ranks one prover's part-8 proof cores with fixed scorers: per-problem ROC AUC, recall@k and whole-core coverage@k.
# usage: rank_transfer.py SOURCE OUTPUT.csv THREADS CHECKPOINT...   SOURCE is a rows.jsonl or a joined dataset directory.
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path[:0] = ['/home/msf43/connections/packages', '/home/msf43/connections/src']
from axiom_prediction.data import axiom_training_example_from_json
from axiom_prediction.model import scripted_logits
from axiom_prediction.split import ProblemSplit

cutoffs = (16, 64, 256)
fields = ['problem', 'checkpoint', 'axioms', 'core', 'auc', 'worst_rank', *[f'{name}{k}' for k in cutoffs for name in ('recall', 'full')]]

def auc(scores, labels):
    # Mann-Whitney with average ranks for ties.
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        same = scores == value
        ranks[same] = ranks[same].mean()

    positives = labels > 0
    return (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * (~positives).sum())

def part8_examples(source):
    split = ProblemSplit(split = 10, parts = [8], scheme = 5)
    if source.is_dir():
        from axiom_prediction.joined import load_joined
        held = {}
        for example in load_joined(source, max_bytes = int(10e6))[0]:
            if split.contains(example.problem_path):
                held.setdefault(Path(example.problem_path).name, example)

        return held

    held = {}
    for line in source.open():
        if len(line) <= 10e6:
            row = json.loads(line)
            if 'example' in row:
                held[row['problem']] = axiom_training_example_from_json(row['example'])

    return held

def metrics(scores, labels):
    order = np.argsort(-scores, kind = 'stable')
    position = np.empty(len(scores), dtype = int)
    position[order] = np.arange(1, len(scores) + 1)
    out = {'axioms': len(labels), 'core': int(labels.sum()), 'auc': round(float(auc(scores, labels)), 4), 'worst_rank': int(position[labels > 0].max())}
    for k in cutoffs:
        if len(labels) > k:
            out[f'recall{k}'] = round(float(labels[order[:k]].sum() / labels.sum()), 4)
            out[f'full{k}'] = int(out['worst_rank'] <= k)

    return out

def main():
    source, output, threads, checkpoints = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    torch.set_num_threads(threads)
    examples = part8_examples(source)
    # LIMIT keeps a login-node check short.
    examples = dict(list(examples.items())[:int(os.environ.get('LIMIT', 10 ** 9))])
    done = {(r['problem'], r['checkpoint']) for r in csv.DictReader(output.open())} if output.exists() else set()
    print(f'{source}: {len(examples)} part-8 problems, {len(done)} rows done', flush = True)
    with output.open('a', newline = '') as handle:
        writer = csv.DictWriter(handle, fieldnames = fields)
        if not done:
            writer.writeheader()

        for path in checkpoints:
            module = torch.jit.load(path, map_location = 'cpu')
            name = '-'.join(Path(path).parts[-3:]) if 'epoch-' in path else Path(path).name
            with torch.no_grad():
                for problem, example in examples.items():
                    labels = np.array(example.labels)
                    if (problem, name) in done or not 0 < labels.sum() < len(labels):
                        continue

                    scores = scripted_logits(module, example.graph).numpy()
                    writer.writerow({'problem': problem, 'checkpoint': name, **metrics(scores, labels)})
                    handle.flush()

            print(f'{name} done', flush = True)

if __name__ == '__main__':
    main()
