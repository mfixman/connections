"""Shared, adaptively batched inference for parallel run/evaluate workloads."""
from contextlib import contextmanager
from functools import lru_cache
import gc
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import queue
import secrets
import threading
import time

import torch

from .graph import build_axiom_graph, collate_axiom_graphs
from .logs import log


class AdaptiveBatches:
    """Double after success; bisect the successful/failed interval after OOM.

    Graph sizes vary, so every batch (including singletons) is checked again.
    A failed batch never commits partial output.
    """
    def __init__(self, maximum, initial=1):
        self.maximum = max(1, maximum)
        self.size = min(self.maximum, max(1, initial))
        self.good = 0
        self.bad = None

    def success(self, count):
        if count < self.size:
            return
        self.good = count
        self.size = min(self.maximum, count * 2 if self.bad is None
                        else max(count, (count + self.bad) // 2))
        log(f"GPU batching: {count} succeeded; next target {self.size}")

    def oom(self, count):
        self.bad = count
        if self.good >= count:
            self.good = 0
        self.size = max(1, (self.good + count) // 2)
        log(f"GPU batching: OOM at {count}; retry target {self.size}")


def predict_graphs(model, graphs):
    with torch.inference_mode():
        batch = collate_axiom_graphs([(graph, None) for graph in graphs])
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model produced non-finite logits")
        values = torch.sigmoid(logits).cpu().tolist()
    result = []
    offset = 0
    for graph in graphs:
        end = offset + len(graph.axiom_clause_ids)
        result.append(values[offset:end])
        offset = end
    if offset != len(values):
        raise ValueError("prediction count does not match batch axiom count")
    return result


def adaptive_predictions(model, graphs, tuner, *, tolerate_singleton_oom=False):
    offset = 0
    while offset < len(graphs):
        count = min(tuner.size, len(graphs) - offset)
        failed = False
        try:
            values = predict_graphs(model, graphs[offset:offset + count])
        except torch.cuda.OutOfMemoryError:
            if count == 1 and not tolerate_singleton_oom:
                raise
            failed = True
        # Leave the exception scope before collecting: its traceback holds tensors.
        if failed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if count == 1:
                yield RuntimeError("one problem exceeds GPU memory even at batch size 1")
                offset += 1
                tuner.good, tuner.bad, tuner.size = 0, None, 1
            else:
                tuner.oom(count)
            continue
        tuner.success(count)
        yield from values
        offset += count


class InferenceService:
    def __init__(self, checkpoint, device, maximum):
        from .model import AxiomPredictor
        torch.set_num_threads(1)
        self.predictor = AxiomPredictor.load(checkpoint, device=device)
        self.tuner = AdaptiveBatches(maximum)
        self.requests = queue.Queue(maxsize=maximum)
        threading.Thread(target=self.serve, daemon=True).start()

    def metadata(self):
        return self.predictor.training_config

    def predict(self, graph):
        done = threading.Event()
        output = []
        self.requests.put((graph, done, output))
        done.wait()
        if isinstance(output[0], Exception):
            raise output[0]
        return output[0]

    def serve(self):
        while True:
            requests = [self.requests.get()]
            deadline = time.monotonic() + 0.05
            while len(requests) < self.tuner.size:
                try:
                    requests.append(self.requests.get(timeout=max(0, deadline-time.monotonic())))
                except queue.Empty:
                    break
            try:
                values = list(adaptive_predictions(self.predictor.model,
                    [r[0] for r in requests], self.tuner, tolerate_singleton_oom=True))
            except Exception as error:
                # Return serializable errors, without retaining CUDA tracebacks.
                values = [RuntimeError(f"shared inference failed: {type(error).__name__}: {error}")] * len(requests)
            if any(isinstance(value, Exception) for value in values):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for (_, done, output), value in zip(requests, values, strict=True):
                output.append(value)
                done.set()


_service = None


def initialize_service(checkpoint, device, maximum):
    global _service
    _service = InferenceService(checkpoint, device, maximum)


def get_service():
    return _service


class InferenceManager(BaseManager):
    pass


InferenceManager.register('predictor', callable=get_service, exposed=('predict', 'metadata'))


@contextmanager
def inference_service(checkpoint, device, maximum):
    key = secrets.token_bytes(32)
    manager = InferenceManager(address=('127.0.0.1', 0), authkey=key,
                               ctx=mp.get_context('spawn'))
    manager.start(initializer=initialize_service, initargs=(checkpoint, device, maximum))
    try:
        log(f"shared inference ready; up to {maximum} CPU proof workers, one GPU model")
        yield manager.address, key.hex()
    finally:
        manager.shutdown()


class SharedPredictor:
    def __init__(self, address, key):
        self.manager = InferenceManager(address=address, authkey=bytes.fromhex(key))
        self.manager.connect()
        self.proxy = self.manager.predictor()
        self.training_config = self.proxy.metadata()

    def predict(self, matrix, *, axiom_clause_ids, conjecture_clause_ids):
        from .model import AxiomPrediction
        graph = build_axiom_graph(matrix, axiom_clause_ids=axiom_clause_ids,
                                  conjecture_clause_ids=conjecture_clause_ids)
        probabilities = self.proxy.predict(graph)
        ranked = sorted(zip(graph.axiom_clause_ids, probabilities, strict=True),
                        key=lambda item: (-item[1], item[0]))
        return tuple(AxiomPrediction(index, str(matrix.clauses[index]), float(value), rank)
                     for rank, (index, value) in enumerate(ranked, start=1))


@lru_cache(maxsize=1)
def shared_predictor(address, key):
    return SharedPredictor(address, key)
