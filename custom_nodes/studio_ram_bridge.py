"""One-job, loopback-only media bridge for the private studio.

Neither input images nor encoded videos are written to a filesystem path.
The ComfyUI process is reaped after each job to release all remaining media.
"""
import ctypes
import io
import re
import resource

import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps
from server import PromptServer

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)


INPUTS = {}
OUTPUTS = {}
TOKEN = re.compile(r"[a-f0-9]{32}\Z")


def checked(request):
    token = request.match_info.get("token", "")
    if not TOKEN.fullmatch(token) or request.remote not in ("127.0.0.1", "::1"):
        raise web.HTTPForbidden()
    return token


@PromptServer.instance.routes.post("/studio-ram/input/{token}")
async def put_input(request):
    token = checked(request)
    if request.content_length is None or request.content_length > 4 * 1024 * 1024:
        raise web.HTTPRequestEntityTooLarge(max_size=4 * 1024 * 1024, actual_size=request.content_length or 0)
    payload = await request.read()
    if not payload or len(payload) > 4 * 1024 * 1024:
        raise web.HTTPBadRequest()
    try:
        with Image.open(io.BytesIO(payload)) as im:
            im.verify()
    except Exception:
        raise web.HTTPBadRequest()
    INPUTS[token] = payload
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/studio-ram/output/{token}")
async def get_output(request):
    token = checked(request)
    payload = OUTPUTS.pop(token, None)
    if payload is None:
        raise web.HTTPNotFound()
    INPUTS.pop(token, None)
    return web.Response(body=payload, content_type="video/mp4", headers={"Cache-Control": "no-store"})


class StudioRAMLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"token": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "studio"

    def load(self, token):
        payload = INPUTS.pop(token, None)
        if payload is None:
            raise ValueError("RAM input missing")
        with Image.open(io.BytesIO(payload)) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            array = np.array(image, dtype=np.float32, copy=True) / 255.0
        return (torch.from_numpy(array).unsqueeze(0),)


class StudioRAMSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "token": ("STRING", {"default": ""})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "studio"

    def save(self, video, token):
        if not TOKEN.fullmatch(token):
            raise ValueError("Invalid RAM output token")
        output = io.BytesIO()
        video.save_to(output, format="mp4", codec="h264")
        OUTPUTS[token] = output.getvalue()
        return {"ui": {}}


NODE_CLASS_MAPPINGS = {
    "StudioRAMLoadImage": StudioRAMLoadImage,
    "StudioRAMSaveVideo": StudioRAMSaveVideo,
}
