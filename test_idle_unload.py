"""Verify that image workers are lazy and release resident model processes after idle timeout."""
import threading

from gpu_workers import GpuRunner


class FakeProcess:
    def __init__(self):
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        pass

    def kill(self):
        self.alive = False


class FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


runner = GpuRunner(0)
assert not runner.resident()
runner.process = process = FakeProcess()
runner.connection = connection = FakeConnection()
runner.cancel_event = threading.Event()
runner.last_activity = 100.0

assert not runner.stop_if_idle(300, now=399.99)
assert runner.resident() and not process.terminated
assert runner.stop_if_idle(300, now=400.0)
assert process.terminated and connection.closed and not runner.resident()
assert not runner.stop_if_idle(300, now=1000.0)

print("PASS: lazy worker state and exact idle-unload threshold")
