"""Exercise GPU preference and global stop controls without loading a model."""
import asyncio
import io
import threading
import uuid

import httpx
from PIL import Image

import app as studio


def ident():
    return str(uuid.uuid4())


async def main():
    with io.BytesIO() as buffer:
        Image.new('RGB', (512, 512), 'white').save(buffer, 'PNG')
        png = buffer.getvalue()
    calls = []

    class Runner:
        def __init__(self, gpu):
            self.gpu = gpu
            self.cancelled = threading.Event()

        def start(self):
            pass

        def cancel(self):
            self.cancelled.set()

        def stop(self):
            self.cancel()

        def run(self, params, source, progress):
            self.cancelled.clear()
            calls.append((self.gpu, params[0]))
            progress('sampling', 1, 2)
            if params[0] == 'hold':
                while not self.cancelled.wait(.01):
                    progress('sampling', 1, 2)
                raise RuntimeError('Cancelled')
            progress('sampling', 2, 2)
            return png, dict(gpu=self.gpu, seed=1, seconds=.01)

    studio.RUNNER_FACTORY = Runner
    async with studio.app.router.lifespan_context(studio.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio.app),
                                     base_url='http://localhost:7860') as client:
            owners = [ident() for _ in range(4)]

            async def post(owner, path, data):
                response = await client.post(path, json=data, headers={
                    'Origin': 'http://localhost:7860',
                    'X-Studio-Tab': owner, 'X-Studio-Page': ident()})
                assert response.status_code == 200, (path, response.status_code, response.text)
                return response.json()

            async def start(owner, prompt):
                job_id, batch_id = ident(), ident()
                await post(owner, '/api/batches/start', dict(id=batch_id, mode='text',
                    prompt=prompt, width=512, height=512, seed=1, steps=2,
                    items=[dict(id=job_id, label=prompt)]))
                return job_id

            async def wait(predicate):
                for _ in range(400):
                    if predicate():
                        return
                    await asyncio.sleep(.01)
                raise AssertionError('Timed out')

            first = await start(owners[0], 'hold')
            await wait(lambda: studio.JOBS[first]['stage'] == 'sampling')
            assert studio.JOBS[first]['gpu'] == 0
            second = await start(owners[1], 'hold')
            await wait(lambda: studio.JOBS[second]['stage'] == 'sampling')
            assert studio.JOBS[second]['gpu'] == 1
            queued = await start(owners[2], 'queued')
            listed = (await post(owners[3], '/api/jobs/active', {}))['jobs']
            assert [job['id'] for job in listed] == [first, second, queued]
            await post(owners[3], '/api/jobs/stop', {'id': queued})
            assert studio.JOBS[queued]['stage'] == 'error' and studio.JOBS[queued]['source'] is None
            await post(owners[3], '/api/jobs/stop', {'id': first})
            await wait(lambda: studio.ACTIVE_JOBS[0] is None)
            assert studio.JOBS[first]['stage'] == 'error'
            replacement = await start(owners[2], 'fast')
            await wait(lambda: studio.JOBS[replacement]['stage'] == 'done')
            assert studio.JOBS[replacement]['gpu'] == 0
            assert (1, 'queued') not in calls and (0, 'queued') not in calls
            await post(owners[3], '/api/jobs/stop', {'id': second})
            await wait(lambda: studio.ACTIVE_JOBS[1] is None)
            assert studio.JOBS[second]['stage'] == 'error'
            assert (await post(owners[3], '/api/jobs/active', {}))['total'] == 0
            for owner in owners:
                await post(owner, '/api/tab/clear', {})
    print('PASS: GPU 0 first; GPU 1 parallel; all jobs visible; queued and running stops; GPU 0 reused')


asyncio.run(main())
