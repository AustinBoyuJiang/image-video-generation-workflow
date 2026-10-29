"""Check the image-to-video API contract without loading model weights."""
import asyncio
import base64
import uuid

import httpx

import app as studio


def ident():
    return str(uuid.uuid4())


class FakeImageRunner:
    def __init__(self, gpu):
        self.gpu = gpu

    def start(self):
        pass

    def stop(self):
        pass

    def cancel(self):
        pass


class FakeVideoRunner:
    def __init__(self, gpu):
        self.gpu = gpu

    def cancel(self):
        pass

    def run(self, params, source, progress):
        assert source == b"fake-jpeg-content"
        assert params[5] == 30
        assert params[6] == "base"
        progress("sampling", 1, 300)
        result = b"\x00\x00\x00\x18ftypisom" + b"fake-video"
        return result, dict(gpu=self.gpu, seed=7, seconds=.01, width=512,
                            height=512, mode="video", duration=30, segments=6)


async def main():
    studio.RUNNER_FACTORY = FakeImageRunner
    studio.H3Runner = FakeVideoRunner
    studio.models_ready = lambda profile="fast": profile in ("fast", "base")
    owner, other, page = ident(), ident(), ident()
    async with studio.app.router.lifespan_context(studio.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio.app),
                                     base_url="http://localhost:7860") as client:
            def headers(tab):
                return {"Origin": "http://localhost:7860", "X-Studio-Tab": tab,
                        "X-Studio-Page": page}

            async def post(path, data, tab=owner, status=200):
                response = await client.post(path, json=data, headers=headers(tab))
                assert response.status_code == status, (path, response.text)
                return response

            upload_id, batch_id, job_id = ident(), ident(), ident()
            source = b"fake-jpeg-content"
            await post("/api/upload/start", {"id": upload_id, "total": len(source)})
            await post("/api/upload/chunk", {"id": upload_id, "index": 0,
                                             "chunk": base64.b64encode(source).decode()})
            spec = dict(id=batch_id, mode="video", prompt="Slow camera pan",
                        width=512, height=512, seed=7, steps=40, duration=30,
                        video_profile="base",
                        items=[dict(id=job_id, label="frame.jpg", upload_id=upload_id)])
            await post("/api/batches/start", spec)
            for _ in range(100):
                batch = (await post("/api/batches/status", {"id": batch_id})).json()
                if batch["stage"] == "done":
                    break
                await asyncio.sleep(.01)
            else:
                raise AssertionError("Video job did not finish")
            assert batch["completed"] == 1 and batch["spec"]["duration"] == 30
            assert batch["spec"]["video_profile"] == "base"
            result = await post("/api/jobs/result", {"id": job_id})
            assert result.headers["content-type"] == "video/mp4"
            assert result.content.startswith(b"\x00\x00\x00\x18ftyp")
            await post("/api/jobs/result", {"id": job_id}, other, 404)
            await post("/api/tab/clear", {})
            assert not studio.JOBS and not studio.BATCHES
    print("PASS: 30-second image-to-video job, MP4 delivery, and tab isolation")


asyncio.run(main())
