# Log lines that describe a training dataset and each finished epoch.

from collections import Counter
import math
import re
import statistics

from .logs import log

def spread(values):
    if not values:
        return "n/a"

    return f"min {min(values):.3g}, median {statistics.median(values):.3g}, mean {statistics.fmean(values):.3g}, max {max(values):.3g}"

# `TPTPParseError: [CODE] details` becomes `TPTPParseError [CODE]`, so that similar failures count together.
def outcome_kind(outcome):
    head, _, rest = outcome.partition(": ")
    code = re.match(r"\s*(\[[A-Z0-9_]+\])", rest)
    return head if code is None else f"{head} {code.group(1)}"

def log_dataset(examples, skipped, batch_size):
    sizes = [len(example.labels) for example in examples]
    cores = [int(sum(example.labels)) for example in examples]
    nodes = [sum(map(len, example.graph.graph.nodes.values())) for example in examples]
    edges = [sum(map(len, example.graph.graph.edges.values())) for example in examples]
    duplicates = len(examples) - len({example.problem_path for example in examples})
    rate = f"{100 * sum(cores) / sum(sizes):.2f}%" if sum(sizes) else "n/a"
    log(
        f"dataset: {len(examples)} examples ({duplicates} repeat a problem with another core, {len(skipped)} problems skipped), "
        f"{sum(sizes)} axiom clauses, {sum(cores)} in SAT cores ({rate}), {math.ceil(len(examples) / batch_size)} batches per epoch"
    )

    outcomes = Counter(outcome_kind(item.get("outcome", "")) for item in skipped)
    if outcomes:
        log("skipped outcomes: " + ", ".join(f"{kind} x{count}" for kind, count in outcomes.most_common()))

    log(f"axiom clauses per problem: {spread(sizes)}")
    log(f"SAT-core clauses per problem: {spread(cores)}; core fraction {spread([c / n for c, n in zip(cores, sizes) if n])}")
    log(f"graph nodes per problem: {spread(nodes)}; edges per problem: {spread(edges)}")

def metric_text(metrics):
    names = "bce roc_auc average_precision macro_average_precision macro_recall_at_1 macro_recall_at_5 macro_recall_at_10".split()
    return ", ".join(
        f"{name} {'n/a' if metrics.get(name) is None else f'{metrics[name]:.4f}'}"
        for name in names
        if name in metrics
    )

def log_epoch(epoch, epochs, loss, duration_s, norms, train_metrics):
    norm = f"{sum(norms) / len(norms):.4f}" if norms else "n/a"
    loss_text = "n/a" if loss is None else f"{loss:.6f}"
    suffix = "" if train_metrics is None else f"; train-set {metric_text(train_metrics)}"
    log(f"epoch {epoch}/{epochs}: loss {loss_text}, mean grad norm {norm}, {len(norms)} batches in {duration_s:.2f}s{suffix}")
