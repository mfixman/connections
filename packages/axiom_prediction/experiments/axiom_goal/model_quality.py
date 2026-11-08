# Held-out ranking quality on part 8: per-problem ROC AUC and recall@k of unsat-core axioms, per checkpoint.
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import paths

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'AUC and recall@k of checkpoints on the part-8 examples')
    parser.add_argument('dataset', type = Path, help = 'joined dataset directory')
    parser.add_argument('checkpoints', nargs = '+')
    parser.add_argument('--threads', type = int, default = 1)
    parser.add_argument('--output', type = Path, default = paths.goal / 'model-quality-part8.json')
    parser.add_argument('--union', action = 'store_true', help = 'score against the union of cores over all seeds (MUSHyperNet-style labels); appends -union to the output name')
    parser.add_argument('--connections', type = Path, default = paths.connections)
    return parser.parse_args()

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

def held_out(examples, part):
    from axiom_prediction.split import ProblemSplit

    split = ProblemSplit(split = 10, parts = [part], scheme = 5)
    held, union = {}, {}
    for example in examples:
        if not split.contains(example.problem_path):
            continue

        held.setdefault(example.problem_path, example)
        merged = union.setdefault(example.problem_path, np.zeros(len(example.labels)))
        if len(example.labels) == len(merged):
            np.maximum(merged, example.labels, out = merged)

    return held, union

def quality(module, held, labels_of):
    from axiom_prediction.model import scripted_logits

    aucs, recall = [], {16: [], 64: []}
    with torch.no_grad():
        for problem, example in held.items():
            labels = labels_of(problem, example)
            if not 0 < labels.sum() < len(labels):
                continue

            scores = scripted_logits(module, example.graph).numpy()
            aucs.append(auc(scores, labels))
            top = np.argsort(-scores)
            for k in recall:
                if len(labels) > k:
                    recall[k].append(labels[top[:k]].sum() / labels.sum())

    return {'problems': len(aucs), 'auc': float(np.mean(aucs)), **{f'recall@{k}': float(np.mean(v)) for k, v in recall.items()}, **{f'n>{k}': len(v) for k, v in recall.items()}}

def main():
    args = parse_arguments()
    paths.add_connections(args.connections)
    from axiom_prediction.joined import load_joined

    torch.set_num_threads(args.threads)
    examples, _, _ = load_joined(args.dataset, max_bytes = int(10e6))
    held, union = held_out(examples, 8)
    output = args.output
    if args.union:
        output = output.with_name(output.name.replace('.json', '-union.json'))
        sizes = [(len(v), v.sum()) for v in union.values()]
        print('mean axioms %.1f, mean union-core size %.1f' % tuple(np.mean(sizes, axis = 0)), flush = True)

    labels_of = (lambda problem, example: union[problem]) if args.union else (lambda problem, example: np.array(example.labels))
    print(f'{len(held)} part-8 problems (one example each)', flush = True)
    scores = {}
    for path in args.checkpoints:
        scores[path] = quality(torch.jit.load(path, map_location = 'cpu'), held, labels_of)
        print(path.split('/')[-3], path.split('/')[-1], scores[path], flush = True)
        json.dump(scores, open(output, 'w'), indent = 1)

main()
