from contextlib import contextmanager
from functools import wraps
import signal
import threading

default_step_limit = 1_000_000
default_timeout_s = 120.0
default_collection_step_limit = 1_000_000_000
default_collection_timeout_s = 900.0

class CollectionTimeout(BaseException):
    # Escape parser and policy handlers that catch ordinary exceptions.
    pass

@contextmanager
def wall_clock(duration_s):
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    if duration_s <= 0:
        raise CollectionTimeout

    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("axiom collection cannot replace an active wall-clock alarm")

    def expired(signum, frame):
        raise CollectionTimeout

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, duration_s, 1.0)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

def collection_budget(function):
    @wraps(function)
    def collect(*args, **kwargs):
        duration_s = kwargs.get(
            "timeout_s",
            function.__kwdefaults__["timeout_s"],
        )

        try:
            with wall_clock(duration_s):
                return function(*args, **kwargs)
        except CollectionTimeout:
            return None, "Timeout"

    return collect
