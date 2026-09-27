from contextlib import contextmanager
from functools import wraps
import signal
import threading


class CollectionTimeout(BaseException):
    """Escape parser and policy handlers that catch ordinary exceptions."""


@contextmanager
def wall_clock(seconds):
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return
    if seconds <= 0:
        raise CollectionTimeout
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("axiom collection cannot replace an active wall-clock alarm")

    def expired(_signum, _frame):
        raise CollectionTimeout

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def collection_budget(function):
    @wraps(function)
    def collect(*args, **kwargs):
        seconds = kwargs.get("timeout_seconds", function.__kwdefaults__["timeout_seconds"])
        try:
            with wall_clock(seconds):
                return function(*args, **kwargs)
        except CollectionTimeout:
            return None, "Timeout"

    return collect
