from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
from threading import Event, Thread
from time import monotonic

PROGRESS_SECONDS = 60.0
logger = logging.getLogger("axiom_prediction")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)
latest = (monotonic(), "starting command")

def log(message: str):
    global latest
    latest = (monotonic(), message)
    handler.stream = sys.stderr
    level = logging.WARNING if "warning:" in message.lower() else logging.INFO
    if "axiom-predictor: error:" in message.lower():
        level = logging.ERROR

    logger.log(level, message)

def watch_progress(stopped):
    while not stopped.wait(PROGRESS_SECONDS):
        updated, message = latest
        elapsed = monotonic() - updated
        if elapsed >= PROGRESS_SECONDS:
            logger.info("No new progress for %.0fs; last update: %s", elapsed, message)

@contextmanager
def monitor_progress():
    stopped = Event()
    thread = Thread(target = watch_progress, args = (stopped,), daemon = True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()

def progress(items, label, total = None):
    started = last_report = monotonic()
    count = 0
    target = "" if total is None else f"/{total}"
    log(f"{label}: starting" + ("" if total is None else f" ({total} items)"))
    for item in items:
        yield item
        count += 1
        now = monotonic()
        if now - last_report >= PROGRESS_SECONDS:
            log(f"{label}: {count}{target} completed in {now - started:.0f}s")
            last_report = now

    log(f"{label}: completed {count}{target} in {monotonic() - started:.1f}s")

__all__ = ["log", "monitor_progress", "progress"]
