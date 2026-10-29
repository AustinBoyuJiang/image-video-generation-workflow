"""Independent spawned CUDA processes. All prompts and media travel only over RAM IPC."""
import multiprocessing
import os
import threading
import time


def _child(connection, cancel, gpu):
    import ctypes
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
    os.environ['QWEN_GPU'] = str(gpu)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    from engine import generate
    while True:
        try:
            message = connection.recv()
        except EOFError:
            break
        if message is None:
            break
        params, source = message
        def progress(stage, step=0, total=0):
            if cancel.is_set():
                raise RuntimeError('Cancelled')
            connection.send(('progress', stage, step, total))
        try:
            progress('loading')
            payload, stats = generate(*params[:5], source, 0.6, progress=progress)
            stats['gpu'] = gpu
            if cancel.is_set():
                connection.send(('error', 'Cancelled'))
            else:
                connection.send(('result', payload, stats))
        except Exception as error:
            # Never serialize error messages or tracebacks containing user input.
            connection.send(('error', type(error).__name__))
        finally:
            message = params = source = payload = stats = None


class GpuRunner:
    def __init__(self, gpu):
        self.gpu = gpu
        self.context = multiprocessing.get_context('spawn')
        self.lock = threading.Lock()
        self.process = self.connection = self.cancel_event = None

    def start(self):
        parent, child = self.context.Pipe()
        self.cancel_event = self.context.Event()
        self.process = self.context.Process(target=_child, args=(child, self.cancel_event, self.gpu), daemon=True,
                                            name=f'studio-gpu-{self.gpu}')
        self.process.start()
        child.close()
        self.connection = parent

    def cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()

    def stop(self):
        self.cancel()
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def run(self, params, source, progress):
        with self.lock:
            if self.process is None or not self.process.is_alive():
                self.stop()
                self.start()
            self.cancel_event.clear()
            progress('loading')
            cancelled = False
            started = time.monotonic()
            try:
                self.connection.send((params, source))
                while True:
                    if not self.connection.poll(1):
                        if not self.process.is_alive():
                            raise RuntimeError('GPU worker exited')
                        if time.monotonic() - started > 1800:
                            raise TimeoutError('GPU worker timeout')
                        continue
                    message = self.connection.recv()
                    if message[0] == 'progress':
                        if not cancelled:
                            try:
                                progress(*message[1:])
                            except Exception:
                                cancelled = True
                                self.cancel()
                    elif message[0] == 'result':
                        if cancelled:
                            raise RuntimeError('Cancelled')
                        return message[1], message[2]
                    else:
                        raise RuntimeError('GPU worker failed')
            except (EOFError, BrokenPipeError, OSError, TimeoutError):
                self.stop()
                raise RuntimeError('GPU worker unavailable') from None
            finally:
                params = source = message = None
