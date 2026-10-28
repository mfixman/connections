# Validation runs on their own cores in a thread, so training never waits for proof search.

from __future__ import annotations

import os
import threading

from .logs import log
from .validation_run import run_baseline, run_validation

def allowed_cpus(count):
    return os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(count))

# Linux pins the calling thread only; elsewhere this is a no-op.
def pin(cpus):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)

def split_cpus(count: int) -> tuple[set[int], set[int]]:
    cpus = sorted(allowed_cpus(count))[:count]
    if len(cpus) < 2:
        return set(cpus), set(cpus)

    return {cpus[0]}, set(cpus[1:])

class BackgroundValidation:
    def __init__(self, problems, configs, tracker, *, count, tptp_root):
        self.problems = problems
        self.configs = configs
        self.tracker = tracker
        self.tptp_root = tptp_root

        self.original_cpus = allowed_cpus(count)
        training_cpus, self.cpus = split_cpus(count)
        pin(training_cpus)
        log(f"validation: training on CPUs {sorted(training_cpus)}, proof search on {len(self.cpus)} others")

        self.base_metrics = {}
        self.pending = None
        self.closing = False
        self.stop = threading.Event()
        self.condition = threading.Condition()

        self.thread = threading.Thread(target = self.loop, name = "validation", daemon = True)
        self.thread.start()

    def submit(self, epoch, checkpoint):
        with self.condition:
            if self.pending is not None:
                log(f"validation: skipping epoch {self.pending[0]}; proof search is still busy")

            self.pending = (epoch, checkpoint)
            self.condition.notify()

    # Waits for the latest submitted epoch; aborting stops at the next finished problem.
    def finish(self, abort = False):
        if abort:
            self.stop.set()

        with self.condition:
            self.closing = True
            self.condition.notify()

        self.thread.join()
        pin(self.original_cpus)

    def loop(self):
        pin(self.cpus)
        options = dict(tptp_root = self.tptp_root, num_workers = len(self.cpus), stop = self.stop)
        try:
            self.base_metrics = run_baseline(self.problems, self.configs, **options) or {}
            self.publish(self.base_metrics, summary = True)
            while (task := self.next_task()) is not None:
                epoch, checkpoint = task
                metrics = run_validation(self.problems, self.configs, checkpoint, **options)
                if metrics is None:
                    return

                self.publish({"epoch": epoch, **metrics, **self.base_metrics})
        except Exception as error:
            log(f"warning: validation stopped after an error ({type(error).__name__}: {error}); training continues")

    def next_task(self):
        with self.condition:
            while self.pending is None and not self.closing:
                self.condition.wait()

            task, self.pending = self.pending, None
            return None if self.stop.is_set() else task

    def publish(self, metrics, summary = False):
        if self.tracker is None:
            return

        payload = {k if k == "epoch" else f"evaluation/{k}": v for k, v in metrics.items()}
        if summary:
            self.tracker.run.summary.update(payload)
        else:
            self.tracker.run.log(payload)
