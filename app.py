"""Loopback-only, passwordless studio with tab-scoped, RAM-only FIFO jobs."""
import asyncio
import base64
import ctypes
import io
import logging
import math
import os
import re
import resource
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent
WEB_ROOT = Path(os.environ.get("STUDIO_WEB_ROOT", ROOT / "web"))
PORT = int(os.environ.get("Z_IMAGE_PORT", "7860"))
VERSION = "1.1.0"
TABS, JOBS, UPLOADS, BATCHES = {}, {}, {}, {}
QUEUE = asyncio.Queue()
GPU_QUEUES = {}
GPU_RESERVED = set()
GPU_AVAILABLE = asyncio.Event()
ACTIVE_JOBS = {}
GPU_RUNNERS = {}
VIDEO_RUNNERS = {}
GPU_IDS = tuple(dict.fromkeys(int(x) for x in os.environ.get("QWEN_GPUS", "0,1").split(",")))
from gpu_workers import GpuRunner
from h3_runner import H3Runner, models_ready
RUNNER_FACTORY = GpuRunner
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{24,80}\Z")
MAX_QUEUE = 64
MAX_BATCH = 20
TAB_LEASE = 3600
CLOSE_GRACE = 180
CHUNK_SIZE = 49152


def valid_id(value):
    return isinstance(value, str) and ID_PATTERN.fullmatch(value) is not None


def allowed_origin(value):
    try:
        p = urlsplit(value or "")
        return (p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1")
                and p.port is not None and 1 <= p.port <= 65535 and not p.path
                and not p.query and not p.fragment and p.username is None and p.password is None)
    except ValueError:
        return False


async def body(request, limit=98304):
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(415, "Use a JSON request.")
    import json
    data = bytearray()
    try:
        async with asyncio.timeout(15):
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > limit:
                    raise HTTPException(413, "Request body is too large.")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except TimeoutError:
        raise HTTPException(408, "Upload connection interrupted. It is safe to retry.") from None
    except (ValueError, UnicodeError):
        raise HTTPException(400, "Invalid request format.") from None
    finally:
        data.clear()


def wipe_upload(ident):
    upload = UPLOADS.pop(ident, None)
    if upload:
        upload["chunks"].clear()


def forget_job(ident):
    job = JOBS.pop(ident, None)
    if job:
        job["cancelled"] = True
        job["result"] = None
        job["delivery"] = None
        job["source"] = None
        job["params"] = None
        gpu = job.get("gpu")
        if ACTIVE_JOBS.get(gpu) == ident and gpu in GPU_RUNNERS:
            (VIDEO_RUNNERS[gpu] if job.get("mode") == "video" else GPU_RUNNERS[gpu]).cancel()


def stop_job(job):
    if job["stage"] in ("done", "error", "stopping"):
        return False
    gpu = job.get("gpu")
    running = ACTIVE_JOBS.get(gpu) == job["id"] and gpu in GPU_RUNNERS
    job.update(cancelled=True, stage="stopping" if running else "error",
               error=None if running else "Stopped by user.",
               finished=None if running else time.monotonic(),
               source=None, params=None, result=None, delivery=None)
    if running:
        (VIDEO_RUNNERS[gpu] if job.get("mode") == "video" else GPU_RUNNERS[gpu]).cancel()
    return True


def forget_batch(ident):
    batch = BATCHES.pop(ident, None)
    if batch:
        for job_id in batch["jobs"]:
            forget_job(job_id)
        batch["spec"].clear()


def clear_tab(ident):
    tab = TABS.get(ident)
    if tab:
        forget_batch(tab.get("job_id"))
        tab["job_id"] = None
    for uid, upload in list(UPLOADS.items()):
        if upload["owner"] == ident:
            wipe_upload(uid)


def tab_session(request):
    ident = request.headers.get("x-studio-tab", "")
    page_id = request.headers.get("x-studio-page", "")
    if not valid_id(ident) or not valid_id(page_id):
        raise HTTPException(400, "Invalid tab ID. Reopen the page.")
    tab = TABS.get(ident)
    if tab is None:
        if len(TABS) >= 128:
            raise HTTPException(429, "Too many open tabs.")
        tab = TABS[ident] = dict(job_id=None, page_id=page_id, touched=time.monotonic(), detached=None)
    tab.update(touched=time.monotonic(), page_id=page_id, detached=None)
    return ident, tab


def find_job(owner, ident):
    job = JOBS.get(ident) if isinstance(ident, str) else None
    if job is None or job["owner"] != owner:
        raise HTTPException(404, "Job is unavailable. It may have been cleared or the service restarted.")
    return job


def find_batch(owner, ident):
    batch = BATCHES.get(ident) if isinstance(ident, str) else None
    if batch is None or batch["owner"] != owner:
        raise HTTPException(404, "Batch is unavailable. It may have been cleared or the service restarted.")
    return batch


def view_job(job):
    waiting = [j for j in JOBS.values() if j["stage"] == "queued" and not j["cancelled"]]
    position = next((n + 1 for n, j in enumerate(waiting) if j is job), 0)
    return dict(id=job["id"], batch_id=job["batch_id"], index=job["index"], label=job["label"],
                mode=job.get("mode", "image"),
                stage=job["stage"], step=job["step"], total=job["total"], gpu=job["gpu"],
                position=position, ahead=position - 1 + sum(v is not None for v in ACTIVE_JOBS.values()) if position else 0,
                width=job["width"], height=job["height"],
                seconds=round(time.monotonic() - job["created"], 1), error=job.get("error"),
                stats=job.get("stats"), result_bytes=len(job["delivery"]) if job["delivery"] is not None else 0)


def view_batch(batch):
    jobs = [view_job(JOBS[ident]) for ident in batch["jobs"] if ident in JOBS]
    completed = sum(j["stage"] == "done" for j in jobs)
    failed = sum(j["stage"] == "error" for j in jobs)
    running = sum(j["stage"] not in ("done", "error", "queued") for j in jobs)
    queued = sum(j["stage"] == "queued" for j in jobs)
    stage = "done" if completed + failed == len(jobs) else "running" if running else "queued"
    return dict(id=batch["id"], stage=stage, jobs=jobs, spec=batch["spec"], total=len(jobs),
                completed=completed, failed=failed, running=running, queued=queued,
                seconds=round(time.monotonic() - batch["created"], 1))


def delivery_bytes(png):
    # Lossless WebP changes transport size, never output pixels; no disk files.
    from PIL import Image
    with Image.open(io.BytesIO(png)) as image:
        with io.BytesIO() as buffer:
            image.save(buffer, format="WEBP", lossless=True, method=1)
            return buffer.getvalue()


async def queue_dispatcher():
    """Assign FIFO jobs to GPU 0 first, then to another idle GPU."""
    while True:
        while len(GPU_RESERVED) == len(GPU_IDS):
            GPU_AVAILABLE.clear()
            await GPU_AVAILABLE.wait()
        ident = await QUEUE.get()
        job = JOBS.get(ident)
        if job is None or job["cancelled"]:
            QUEUE.task_done()
            continue
        gpu = min(gpu for gpu in GPU_IDS if gpu not in GPU_RESERVED)
        GPU_RESERVED.add(gpu)
        GPU_QUEUES[gpu].put_nowait(ident)


async def queue_worker(gpu):
    while True:
        ident = await GPU_QUEUES[gpu].get()
        job = JOBS.get(ident)
        try:
            if job is None or job["cancelled"]:
                continue
            ACTIVE_JOBS[gpu] = ident
            job.update(stage="loading", started=time.monotonic(), gpu=gpu)

            def progress(stage, step=0, total=0):
                if job["cancelled"]:
                    raise RuntimeError("Cancelled")
                job.update(stage=stage, step=step, total=total)

            params, source = job["params"], job["source"]
            try:
                if job.get("mode") == "video":
                    await run_in_threadpool(GPU_RUNNERS[gpu].stop)
                    payload, stats = await run_in_threadpool(VIDEO_RUNNERS[gpu].run, params, source, progress)
                else:
                    png, stats = await run_in_threadpool(GPU_RUNNERS[gpu].run, params, source, progress)
                if job["cancelled"]:
                    continue
                job.update(stage="packing", source=None, params=None)
                if job.get("mode") != "video":
                    payload = await run_in_threadpool(delivery_bytes, png)
                if not job["cancelled"]:
                    job.update(stage="done", result=None if job.get("mode") == "video" else png,
                               delivery=payload, stats=stats,
                               finished=time.monotonic())
            except Exception as error:
                if not job["cancelled"]:
                    job.update(stage="error", source=None, params=None,
                               error=(str(error) if job.get("mode") == "video" and str(error).startswith("H3 ") else
                                      "Video generation failed. Please retry." if job.get("mode") == "video" else
                                      "This image failed. You can retry it; other images will continue."),
                               finished=time.monotonic())
            finally:
                params = source = png = payload = None
                job["source"] = job["params"] = None
        finally:
            if job is not None and job["cancelled"] and job["stage"] == "stopping":
                job.update(stage="error", error="Stopped by user.", finished=time.monotonic())
            ACTIVE_JOBS[gpu] = None
            GPU_RESERVED.discard(gpu)
            GPU_AVAILABLE.set()
            GPU_QUEUES[gpu].task_done()
            QUEUE.task_done()
            job = None


def expire(now=None):
    now = time.monotonic() if now is None else now
    for uid, upload in list(UPLOADS.items()):
        if now - upload["touched"] > 900:
            wipe_upload(uid)
    for ident, tab in list(TABS.items()):
        detached = tab["detached"]
        if now - tab["touched"] > TAB_LEASE or (detached is not None and now - detached > CLOSE_GRACE):
            clear_tab(ident)
            TABS.pop(ident, None)


async def cleanup():
    while True:
        await asyncio.sleep(10)
        expire()


@asynccontextmanager
async def lifespan(app):
    import sys
    from memory_handoff import restore_from_environment
    await run_in_threadpool(restore_from_environment, sys.modules[__name__])
    for gpu in GPU_IDS:
        runner = GPU_RUNNERS[gpu] = RUNNER_FACTORY(gpu)
        runner.start()
        VIDEO_RUNNERS[gpu] = H3Runner(gpu)
        ACTIVE_JOBS[gpu] = None
        GPU_QUEUES[gpu] = asyncio.Queue()
    dispatcher = asyncio.create_task(queue_dispatcher())
    workers = [asyncio.create_task(queue_worker(gpu)) for gpu in GPU_IDS]
    cleaner = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        dispatcher.cancel()
        for worker in workers:
            worker.cancel()
        cleaner.cancel()
        await asyncio.gather(dispatcher, *workers, cleaner, return_exceptions=True)
        await asyncio.gather(*(run_in_threadpool(runner.stop) for runner in GPU_RUNNERS.values()))
        for runner in VIDEO_RUNNERS.values():
            runner.cancel()
        GPU_RUNNERS.clear()
        VIDEO_RUNNERS.clear()
        ACTIVE_JOBS.clear()
        GPU_QUEUES.clear()
        GPU_RESERVED.clear()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def privacy(request, call_next):
    if request.method not in ("GET", "HEAD") and not allowed_origin(request.headers.get("origin")):
        response = JSONResponse({"detail": "Request origin is not allowed."}, status_code=403)
    else:
        response = await call_next(request)
    response.headers.update({
        "Cache-Control": "no-store, no-cache, max-age=0, must-revalidate", "Pragma": "no-cache",
        "Expires": "0", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer", "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'"})
    return response


@app.get("/api/health")
async def health():
    return dict(ok=True, version=VERSION, model="Qwen-Image-2.1 + MiniMax-H3", h3_ready=models_ready(),
                h3_base_ready=models_ready("base"), password=False,
                queued=sum(j["stage"] == "queued" for j in JOBS.values()), running=sum(v is not None for v in ACTIVE_JOBS.values()),
                gpus=[dict(gpu=gpu, busy=ACTIVE_JOBS.get(gpu) is not None) for gpu in GPU_IDS])


@app.get("/")
async def index():
    return HTMLResponse((WEB_ROOT / "index.html").read_text())


@app.get("/assets/{name}")
async def asset(name: str):
    if name not in ("app.js", "style.css"):
        raise HTTPException(404)
    return Response((WEB_ROOT / name).read_bytes(),
                    media_type="text/javascript" if name.endswith(".js") else "text/css")


@app.post("/api/tab/state")
async def tab_state(request: Request):
    owner, tab = tab_session(request)
    await body(request, 1024)
    batch = BATCHES.get(tab["job_id"])
    return dict(batch=view_batch(batch) if batch else None)


@app.post("/api/tab/detach")
async def tab_detach(request: Request):
    data = await body(request, 1024)
    tab = TABS.get(data.get("tab_id")) if valid_id(data.get("tab_id")) else None
    # Ignore a delayed unload beacon from the page before a refresh.
    if tab and tab["page_id"] == data.get("page_id"):
        tab["detached"] = time.monotonic()
    return {"ok": True}


@app.post("/api/tab/clear")
async def tab_clear(request: Request):
    owner, tab = tab_session(request)
    await body(request, 1024)
    clear_tab(owner)
    return {"ok": True}


@app.post("/api/upload/start")
async def upload_start(request: Request):
    owner, tab = tab_session(request)
    data = await body(request, 1024)
    ident, total = data.get("id"), data.get("total")
    if not valid_id(ident) or not isinstance(total, int) or not 0 < total <= 4 * 1024 * 1024:
        raise HTTPException(400, "Invalid upload parameters.")
    old = UPLOADS.get(ident)
    if old and (old["owner"] != owner or old["total"] != total):
        raise HTTPException(409, "Upload ID conflict.")
    if not old:
        if sum(upload["owner"] == owner for upload in UPLOADS.values()) >= MAX_BATCH + 1:
            raise HTTPException(429, "Too many pending uploads for this tab.")
        UPLOADS[ident] = dict(owner=owner, total=total, chunks={}, touched=time.monotonic())
    return {"id": ident}


@app.post("/api/upload/chunk")
async def upload_chunk(request: Request):
    owner, tab = tab_session(request)
    data = await body(request)
    upload = UPLOADS.get(data.get("id")) if isinstance(data.get("id"), str) else None
    if upload is None or upload["owner"] != owner:
        raise HTTPException(404, "Upload expired. Select the image again.")
    index, chunk = data.get("index"), data.get("chunk")
    count = (upload["total"] + CHUNK_SIZE - 1) // CHUNK_SIZE
    if not isinstance(index, int) or not 0 <= index < count or not isinstance(chunk, str) or len(chunk) > 65536:
        raise HTTPException(400, "Invalid upload chunk.")
    try:
        decoded = base64.b64decode(chunk, validate=True)
    except ValueError:
        raise HTTPException(400, "Invalid upload chunk.") from None
    expected = min(CHUNK_SIZE, upload["total"] - index * CHUNK_SIZE)
    if len(decoded) != expected:
        raise HTTPException(400, "Invalid upload chunk length.")
    if index in upload["chunks"] and upload["chunks"][index] != decoded:
        raise HTTPException(409, "Upload chunk conflict.")
    upload["chunks"][index] = decoded
    upload["touched"] = time.monotonic()
    return {"received": sum(map(len, upload["chunks"].values()))}


@app.post("/api/batches/start")
async def batch_start(request: Request):
    owner, tab = tab_session(request)
    data = await body(request, 24576)
    ident = data.get("id")
    if not valid_id(ident):
        raise HTTPException(400, "Invalid batch ID.")
    if ident in BATCHES:
        return view_batch(find_batch(owner, ident))
    old = BATCHES.get(tab["job_id"])
    if old and view_batch(old)["stage"] != "done":
        raise HTTPException(409, "This tab already has an active batch. Wait for it or clear this tab.")
    try:
        mode, prompt = data.get("mode", "text"), data.get("prompt", "")
        width, height = int(data.get("width", 1024)), int(data.get("height", 1024))
        seed, steps = int(data.get("seed", -1)), int(data.get("steps", 40))
        duration = int(data.get("duration", 5)) if mode == "video" else None
        video_profile = data.get("video_profile", "fast") if mode == "video" else None
        if mode not in ("text", "image", "video") or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError()
        if any(x < 512 or x > 1536 or x % 16 for x in (width, height)) or not -1 <= seed < 2**32 or not 1 <= steps <= 50:
            raise ValueError()
        if mode == "video" and (not isinstance(data.get("duration", 5), int) or not 1 <= duration <= 30):
            raise ValueError()
        if mode == "video" and video_profile not in ("fast", "base"):
            raise ValueError()
        items = data.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH or (mode in ("text", "video") and len(items) != 1):
            raise ValueError()
        if any(not isinstance(item, dict) or not valid_id(item.get("id")) or item["id"] in JOBS for item in items):
            raise ValueError()
        if len({item["id"] for item in items}) != len(items):
            raise ValueError()
        if any(not isinstance(item.get("label", ""), str) or len(item.get("label", "")) > 160 for item in items):
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid prompt, settings, or batch images (maximum 20).") from None
    if mode == "video" and not models_ready(video_profile):
        raise HTTPException(503, "The selected MiniMax H3 model is not ready. Please retry later.")
    if sum(j["stage"] == "queued" for j in JOBS.values()) + len(items) > MAX_QUEUE:
        raise HTTPException(429, "Queue is full. Retry the batch later.")
    uploads = []
    if mode in ("image", "video"):
        for item in items:
            uid = item.get("upload_id")
            upload = UPLOADS.get(uid) if isinstance(uid, str) else None
            if upload is None or upload["owner"] != owner or sum(map(len, upload["chunks"].values())) != upload["total"]:
                raise HTTPException(400, "An input image has not finished uploading. No jobs were submitted.")
            uploads.append((uid, upload))
        if len({uid for uid, _ in uploads}) != len(uploads):
            raise HTTPException(400, "Each image needs a separate upload ID.")
    sources = [b"".join(upload["chunks"][n] for n in range(len(upload["chunks"]))) for _, upload in uploads]
    if old:
        forget_batch(old["id"])
    now = time.monotonic()
    spec = dict(mode=mode, prompt=prompt, width=width, height=height, seed=seed, steps=steps)
    if mode == "video":spec.update(duration=duration, video_profile=video_profile)
    batch = dict(id=ident, owner=owner, jobs=[], spec=spec, created=now)
    BATCHES[ident] = batch
    tab["job_id"] = ident
    for index, item in enumerate(items):
        job = dict(id=item["id"], batch_id=ident, owner=owner, index=index, mode=mode,
                   label=item.get("label") or (f"Image {index+1}" if mode != "text" else "Generated image"),
                   stage="queued", step=0, total=(50 if video_profile == "base" else 4) * math.ceil(duration / 5) if mode == "video" else steps, gpu=None, width=width, height=height,
                   created=now, started=None, finished=None, cancelled=False, result=None, delivery=None,
                   params=(prompt, width, height, seed, steps) + ((duration, video_profile) if mode == "video" else ()),
                   source=sources[index] if mode != "text" else None)
        JOBS[job["id"]] = job
        batch["jobs"].append(job["id"])
        QUEUE.put_nowait(job["id"])
    for uid, _ in uploads:
        wipe_upload(uid)
    return view_batch(batch)


@app.post("/api/batches/status")
async def batch_status(request: Request):
    owner, tab = tab_session(request)
    data = await body(request, 1024)
    return view_batch(find_batch(owner, data.get("id")))


@app.post("/api/jobs/status")
async def job_status(request: Request):
    owner, tab = tab_session(request)
    data = await body(request, 1024)
    return view_job(find_job(owner, data.get("id")))


@app.post("/api/jobs/active")
async def active_jobs(request: Request):
    tab_session(request)
    await body(request, 1024)
    jobs = [view_job(job) for job in JOBS.values() if job["stage"] not in ("done", "error")]
    return dict(jobs=jobs, total=len(jobs),
                gpus=[dict(gpu=gpu, busy=ACTIVE_JOBS.get(gpu) is not None) for gpu in GPU_IDS])


@app.post("/api/jobs/stop")
async def stop_active_job(request: Request):
    tab_session(request)
    data = await body(request, 1024)
    job = JOBS.get(data.get("id")) if isinstance(data.get("id"), str) else None
    if job is None:
        raise HTTPException(404, "Job has been cleared.")
    if job["stage"] == "done":
        raise HTTPException(409, "Job is already complete.")
    stopped = stop_job(job)
    return dict(ok=True, stopped=stopped, id=job["id"])


@app.post("/api/jobs/result")
async def job_result(request: Request):
    owner, tab = tab_session(request)
    data = await body(request, 1024)
    job = find_job(owner, data.get("id"))
    if job["stage"] != "done":
        raise HTTPException(409, "Result is not ready.")
    payload = job["delivery"]
    return Response(payload, media_type="video/mp4" if job.get("mode") == "video" else "image/webp",
                    headers={"X-Result-Bytes": str(len(payload))})


@app.post("/api/jobs/result-chunk")
async def job_result_chunk(request: Request):
    owner, tab = tab_session(request)
    data = await body(request, 1024)
    job = find_job(owner, data.get("id"))
    if job["stage"] != "done":
        raise HTTPException(409, "Result is not ready.")
    payload, offset = job["delivery"], data.get("offset", 0)
    if not isinstance(offset, int) or offset < 0 or offset >= len(payload):
        raise HTTPException(400, "Invalid result chunk offset.")
    end = min(offset + CHUNK_SIZE, len(payload))
    return dict(chunk=base64.b64encode(payload[offset:end]).decode("ascii"), next=end, total=len(payload))


if __name__ == "__main__":
    import socket
    import uvicorn
    logging.getLogger("uvicorn.access").disabled = True
    sockets = []
    for port in (PORT,):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(2048)
        sockets.append(sock)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, access_log=False,
                            server_header=False, log_level="warning", timeout_graceful_shutdown=3)
    asyncio.run(uvicorn.Server(config).serve(sockets=sockets))
