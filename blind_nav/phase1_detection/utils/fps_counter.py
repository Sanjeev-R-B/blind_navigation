import time

class FPSCounter:
    def __init__(self):
        self._start = time.time()
        self._count = 0
        self._fps = 0.0

    def tick(self):
        self._count += 1
        elapsed = time.time() - self._start
        if elapsed >= 1.0:
            self._fps = self._count / elapsed
            self._count = 0
            self._start = time.time()

    def get(self) -> float:
        return self._fps