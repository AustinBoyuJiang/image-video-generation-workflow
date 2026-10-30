"""Run one MiniMax H3 image-to-video job on a dedicated GPU.

The subprocess and its prompt history are destroyed after each job. Uploaded
images and MP4 output cross only loopback HTTP into in-process RAM stores.
"""
import io
import json
import math
import os
import secrets
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from PIL import Image
from websockets.exceptions import ConnectionClosed

from h3_media import join_segments, last_frame_png


ROOT = Path(__file__).resolve().parent
COMMON_MODEL_FILES = {
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}
MODEL_PROFILES = {
    "fast": {
        "diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "sampler": "res_multistep",
        "steps": 4,
        "label": "H3 Turbo",
    },
    "base": {
        "diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "sampler": "res_multistep",
        "steps": 50,
        "label": "H3-Base",
    },
    "pika": {
        "diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora": "pika_h3_flex_fl2va_v1_bf16.safetensors",
        "sampler": "lcm",
        "sigmas": "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0",
        "steps": 6,
        "label": "Pika H3 Flex",
    },
}
MODEL_FILES = {
    **COMMON_MODEL_FILES,
    "diffusion_models": MODEL_PROFILES["fast"]["diffusion_models"],
    "loras": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
}


def profile_files(profile):
    files = {**COMMON_MODEL_FILES, "diffusion_models": MODEL_PROFILES[profile]["diffusion_models"]}
    if MODEL_PROFILES[profile].get("lora"):
        files["loras"] = MODEL_PROFILES[profile]["lora"]
    return files


def profile_steps(profile):
    return MODEL_PROFILES[profile]["steps"]


def models_ready(profile="fast"):
    if profile not in MODEL_PROFILES:
        return False
    return all((ROOT / "h3-models" / ("vae" if kind == "audio_vae" else kind) / name).is_file()
               for kind, name in profile_files(profile).items())


def ensure_model_links(profile):
    if not models_ready(profile):
        raise RuntimeError("H3 model is not ready. Please retry later.")
    for kind, name in profile_files(profile).items():
        directory = "vae" if kind == "audio_vae" else kind
        target = ROOT / "h3-models" / directory / name
        link = ROOT / "h3-comfy" / "models" / directory / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(target)


def _request(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _canvas(source):
    with Image.open(io.BytesIO(source)) as image:
        w, h = image.size
    ratio = w / h
    # Approximately 0.4 MP; even dimensions and model's 32 px grid.
    width = max(256, round(math.sqrt(400000 * ratio) / 32) * 32)
    height = max(256, round(math.sqrt(400000 / ratio) / 32) * 32)
    return min(width, 1024), min(height, 1024)


def _graph(prompt, source, seed, token, profile):
    width, height = _canvas(source)
    if seed < 0:
        seed = secrets.randbelow(2**32)
    link = lambda node, slot=0: [str(node), slot]
    model = MODEL_PROFILES[profile]
    graph = {
        "1": {"class_type": "StudioRAMLoadImage", "inputs": {"token": token}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": model["diffusion_models"], "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_FILES["text_encoders"], "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FILES["vae"]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FILES["audio_vae"]}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": link(3), "vae": link(4), "prompt": prompt,
                "width": width, "height": height, "length": 124, "first_frame": link(1)}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": model["sampler"]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": link(8), "guider": link(9),
                "sampler": link(10), "sigmas": link(11), "latent_image": link(6, 1)}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": link(12), "vae": link(4)}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": link(12), "vae": link(5)}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": link(13), "fps": 24.0, "audio": link(14),
                "bit_depth": 8, "color_space": "sRGB", "codec": "none"}},
        "16": {"class_type": "StudioRAMSaveVideo", "inputs": {"video": link(15), "token": token}},
    }
    model_link = link(2)
    if model.get("lora"):
        graph["7"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": link(2), "lora_name": model["lora"], "strength_model": 1.0}}
        model_link = link(7)
    graph["9"] = {"class_type": "BasicGuider", "inputs": {"model": model_link, "conditioning": link(6)}}
    graph["11"] = ({"class_type": "ManualSigmas", "inputs": {"sigmas": model["sigmas"]}}
                   if model.get("sigmas") else
                   {"class_type": "BasicScheduler", "inputs": {
                       "model": model_link, "scheduler": "simple", "steps": model["steps"], "denoise": 1.0}})
    return graph, seed, width, height


class H3Runner:
    def __init__(self, gpu):
        self.gpu = gpu
        self.process = None
        self.lock = threading.Lock()
        self.logs = deque(maxlen=120)

    def _drain_logs(self, process):
        for line in process.stdout:
            self.logs.append(line.decode("utf-8", "replace").rstrip()[:500])

    def cancel(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self, params, source, progress):
        with self.lock:
            profile = params[6] if len(params) > 6 else "fast"
            if profile not in MODEL_PROFILES:
                raise ValueError("Unknown H3 model profile.")
            if not models_ready(profile):
                raise RuntimeError("H3 model is not ready. Please retry later.")
            ensure_model_links(profile)
            prompt, _, _, requested_seed, _ = params[:5]
            seconds = int(params[5]) if len(params) > 5 else 5
            if not 1 <= seconds <= 30:
                raise ValueError("Video duration must be between 1 and 30 seconds.")
            segments = math.ceil(seconds / 5)
            steps = MODEL_PROFILES[profile]["steps"]
            total_steps = steps * segments
            seed = secrets.randbelow(2**32) if requested_seed < 0 else requested_seed
            client_id = secrets.token_hex(16)
            width, height = _canvas(source)
            started = time.monotonic()
            port = 17961 + self.gpu
            base = f"http://127.0.0.1:{port}"
            ws = None
            volatile = tempfile.mkdtemp(prefix="h3-", dir="/dev/shm")
            for name in ("input", "output", "temp", "user"):
                Path(volatile, name).mkdir()
            command = [str(ROOT / "h3-venv/bin/python"), str(ROOT / "h3-comfy/main.py"),
                       "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
                       "--disable-metadata", "--lowvram", "--vram-headroom", "5", "--reserve-vram", "4", "--offline",
                       "--database-url", "sqlite:///:memory:",
                       "--input-directory", str(Path(volatile, "input")),
                       "--output-directory", str(Path(volatile, "output")),
                       "--temp-directory", str(Path(volatile, "temp")),
                       "--user-directory", str(Path(volatile, "user"))]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(self.gpu), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            try:
                progress("loading", 0, total_steps)
                self.logs.clear()
                self.process = subprocess.Popen(command, cwd=ROOT / "h3-comfy", env=env,
                                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                threading.Thread(target=self._drain_logs, args=(self.process,), daemon=True).start()
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError("H3 service failed to start.")
                    try:
                        _request(base + "/system_stats", timeout=2)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                else:
                    raise RuntimeError("H3 service startup timed out.")
                try:
                    from websockets.sync.client import connect
                    ws = connect(f"ws://127.0.0.1:{port}/ws?clientId={client_id}", open_timeout=10, proxy=None)
                    ws.recv(timeout=10)  # Initial queue status; socket is ready before submitting.
                except Exception:
                    if ws is not None:
                        ws.close()
                    ws = None
                parts = []
                current_source = source
                for segment in range(segments):
                    token = secrets.token_hex(16)
                    graph, _, width, height = _graph(prompt, current_source, (seed + segment) % (2**32), token, profile)
                    progress("encoding", segment * steps, total_steps)
                    req = urllib.request.Request(base + f"/studio-ram/input/{token}", data=current_source,
                                                 headers={"Content-Type": "application/octet-stream"})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        response.read()
                    submitted = json.loads(_request(base + "/prompt", {"prompt": graph, "client_id": client_id}, timeout=30))
                    if submitted.get("node_errors") or not submitted.get("prompt_id"):
                        raise RuntimeError("H3 workflow validation failed.")
                    prompt_id = submitted["prompt_id"]
                    progress("sampling", segment * steps, total_steps)
                    deadline = time.monotonic() + 7200
                    while time.monotonic() < deadline:
                        if ws is not None:
                            try:
                                message = ws.recv(timeout=2)
                                if isinstance(message, bytes):
                                    continue  # ComfyUI may send an in-memory preview frame.
                                event = json.loads(message)
                                if event.get("type") == "progress" and event.get("data", {}).get("prompt_id") == prompt_id:
                                    data = event["data"]
                                    progress("sampling", segment * steps + int(data.get("value", 0)), total_steps)
                                elif event.get("type") == "executing" and event.get("data", {}).get("prompt_id") == prompt_id:
                                    node = event["data"].get("node")
                                    if node in ("13", "14"):
                                        progress("decoding", segment * steps + steps, total_steps)
                                    elif node in ("15", "16"):
                                        progress("packing", segment * steps + steps, total_steps)
                            except TimeoutError:
                                pass
                            except (OSError, ValueError, ConnectionClosed):
                                ws.close()
                                ws = None
                        else:
                            time.sleep(2)
                        history = json.loads(_request(base + "/history/" + prompt_id, timeout=15))
                        entry = history.get(prompt_id)
                        if entry:
                            status = entry.get("status", {})
                            if status.get("status_str") == "error":
                                raise RuntimeError("H3 generation failed.")
                            if status.get("completed"):
                                break
                        time.sleep(2)
                    else:
                        raise RuntimeError("H3 generation timed out.")
                    part = _request(base + f"/studio-ram/output/{token}", timeout=300)
                    if not part.startswith(b"\x00\x00\x00") or b"ftyp" not in part[:16]:
                        raise RuntimeError("H3 output is not a valid MP4.")
                    parts.append(part)
                    if segment + 1 < segments:
                        current_source = last_frame_png(part)
                progress("packing")
                video = parts[0] if seconds == 5 else join_segments(parts, seconds,
                    lambda *_: progress("packing", total_steps, total_steps))
                return video, {"gpu": self.gpu, "seed": seed, "seconds": round(time.monotonic() - started, 1),
                               "width": width, "height": height, "mode": "video", "duration": seconds,
                               "profile": profile, "model": MODEL_PROFILES[profile]["label"],
                               "segments": segments}
            finally:
                if ws is not None:
                    ws.close()
                self.cancel()
                if self.process is not None:
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=5)
                    self.process = None
                shutil.rmtree(volatile, ignore_errors=True)
