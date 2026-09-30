from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
import time

from .logs import PROGRESS_SECONDS, log

@dataclass
class SearchWorker:
    process: BaseProcess
    pipe: Connection
    problem: str
    started: float
    completed: int = 0

def worker_loop(pipe, function):
    try:
        while True:
            arguments = pipe.recv()
            try:
                result = function(*arguments)
            except BaseException as error:
                result = {
                    "problem": arguments[0],
                    "proved": False,
                    "error": True,
                    "outcome": f"{type(error).__name__}: {error}",
                }

            pipe.send((result, time.monotonic()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        pipe.close()

def stop_worker(worker):
    worker.pipe.close()
    if worker.process.is_alive():
        worker.process.terminate()

    worker.process.join(timeout = 0.2)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join()

    worker.process.close()

def start_worker(function, arguments):
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target = worker_loop, args = (child, function))
    worker = SearchWorker(process, parent, arguments[0], time.monotonic())
    try:
        process.start()
        child.close()
        parent.send(arguments)
    except BaseException:
        child.close()
        parent.close()
        if process.pid is not None:
            stop_worker(worker)

        raise

    return worker

def supervised_results(function, arguments, *, workers, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")

    pending = iter(arguments)
    active = []
    completed = 0
    last_report = time.monotonic()
    log(f"starting proof search with up to {workers} workers; timeout {timeout:g}s per problem")
    try:
        for _ in range(workers):
            args = next(pending, None)
            if args is None:
                break

            active.append(start_worker(function, args))

        while active:
            delay = min(w.started + timeout for w in active) - time.monotonic()
            ready = wait([w.pipe for w in active], timeout = max(0, min(delay, PROGRESS_SECONDS)))
            now = time.monotonic()
            if now - last_report >= PROGRESS_SECONDS:
                oldest = min(active, key = lambda worker: worker.started)
                log(
                    f"proof search: {completed} completed, {len(active)} active; "
                    f"longest running: {oldest.problem} ({now - oldest.started:.0f}s)"
                )

                last_report = now
            for worker in list(active):
                elapsed = time.monotonic() - worker.started
                reusable = False
                if worker.pipe in ready:
                    try:
                        result, finished = worker.pipe.recv()
                        elapsed = finished - worker.started
                        reusable = elapsed < timeout
                        if not reusable:
                            result = {"outcome": "Timeout"}
                    except EOFError:
                        result = {"outcome": "WorkerExited", "error": True}
                elif elapsed >= timeout:
                    result = {"outcome": "Timeout"}
                else:
                    continue

                result.setdefault("problem", worker.problem)
                result.setdefault("proved", False)
                result["seconds"] = elapsed
                worker.completed += 1
                completed += 1
                args = next(pending, None)
                if reusable and args is not None and worker.completed < 25:
                    worker.problem = args[0]
                    worker.started = time.monotonic()
                    worker.pipe.send(args)
                else:
                    stop_worker(worker)
                    active.remove(worker)
                    if args is not None:
                        active.append(start_worker(function, args))

                yield result
    finally:
        for worker in active:
            stop_worker(worker)
