import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed

    def __str__(self) -> str:
        return format_duration(self.elapsed)


@contextmanager
def timed(label: str = "") -> Generator[Timer, None, None]:
    t = Timer().start()
    yield t
    t.stop()
    if label:
        print(f"{label}: {t}")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"
