# Image Video Generation Workflow

A local web workflow for **Qwen-Image-2.1** text-to-image and image editing, plus **MiniMax H3** image-to-video with an input image and prompt. Multiple browser tabs have separate batch IDs and results. Jobs share a FIFO queue across two GPU workers; the **All jobs** panel can stop queued or running jobs.

The repository contains application code, a RAM-only ComfyUI bridge, setup scripts, and exact model file hashes. **Model weights, generated media, user uploads, local IP addresses, and virtual environments are not included.**

## Tested environment

- Ubuntu 24.04 x86_64, Python 3.11, NVIDIA CUDA
- Two RTX 4080 SUPER GPUs (16 GB VRAM each), 188 GiB system RAM
- PyTorch 2.7.0 with CUDA 12.8, Qwen-Image-2.1 BF16 with CPU group offload
- ComfyUI at the commit in `model-lock.json`, MiniMax H3 int8-convrot weights, 4-step Turbo LoRA, and 50-step Base sampling
- Approximately 71 GB of model files; allow at least 100 GB free disk space for weights, packages, and download staging

This hardware profile is tested, not a minimum requirement. A single GPU can be used by setting `QWEN_GPUS=0`; video generation and image generation still need enough GPU and system memory. Long videos are made in five-second segments and may take many minutes. Segment transitions can change subjects or scenery.

## Install

1. Install NVIDIA drivers, Python 3.11, Git, and a working CUDA runtime. Ensure `python3.11` can create virtual environments.
2. Clone this repository and run:

   ```bash
   ./setup.sh
   .venv/bin/python download_models.py
   QWEN_GPUS=0,1 ./start.sh
   ```

3. Open `http://127.0.0.1:7860/` **on the model host**. The API only listens on loopback. For access from your own computer, use an SSH tunnel with a free local port:

   ```bash
   ssh -N -L 7860:127.0.0.1:7860 user@model-host
   ```

   Then open `http://127.0.0.1:7860/` on that computer. If local port 7860 is occupied, choose another local port in both the tunnel command and browser URL.

`setup.sh` installs two virtual environments, checks out the pinned ComfyUI commit, and installs the custom RAM bridge. It uses PyTorch CUDA 12.8 wheels by default. Set `PYTHON_BIN` or `TORCH_INDEX_URL` before running it if your platform needs different values; changing the Torch build may affect H3 compatibility.

`download_models.py` downloads only the Qwen components and five H3 weight files used by this workflow. Model repositories and revisions are pinned in `model-lock.json`. The script verifies every weight against its expected byte size and SHA256 digest. It can be rerun after an interrupted download:

```bash
.venv/bin/python download_models.py --model qwen
.venv/bin/python download_models.py --model h3
.venv/bin/python download_models.py --verify-only
```

The models are downloaded directly from their publishers: [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) and [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). Review the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) and [MiniMax H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before use. Qwen's research license limits use to non-commercial research or evaluation unless you obtain a separate commercial license. This repository does not grant rights to either model's weights.

## Use

- **Text to image:** enter a prompt. One image is generated per job.
- **Image editing:** select up to 20 input images and enter one prompt, applied separately to each image.
- **Image to video:** select one first-frame image, enter a motion prompt, choose 1–30 seconds at 24 fps, and select H3 Turbo or H3-Base. Turbo uses the four-step LoRA. H3-Base removes the Turbo LoRA and runs the official 50-step base sampler. Both modes use ComfyUI's inference-equivalent AdaLN-pruned int8-convrot weight.
- **New tab:** open an independent workspace. Refreshing a tab recovers its current batch ID. Duplicating a tab gives the copy a new ID.
- **All jobs:** see queue and GPU status and stop individual jobs.

Qwen images are delivered as lossless WebP and H3 videos as MP4. The browser stores only opaque tab and batch IDs in the URL fragment and `sessionStorage`. Inputs and results are held in process memory; H3 uses `/dev/shm` for its temporary ComfyUI directories. Running jobs continue if the page disconnects, and closed tabs retain completed batches for 24 hours. **Clear this tab** removes that tab's batch immediately. The application has no account system and should not be exposed directly to a public network.

## Verify

The queue tests use fake GPU workers and do not download models or create media files:

```bash
.venv/bin/python test_queue.py
.venv/bin/python test_task_control.py
.venv/bin/python test_video_queue.py
curl http://127.0.0.1:7860/api/health
```

A full model smoke test requires the downloaded weights and compatible CUDA hardware. The tested host has run both image generation and H3 video generation; the portable setup script has not been exercised from a blank machine.

## License

The application code in this repository is under the MIT License. PyTorch, Diffusers, ComfyUI, and other dependencies have their own licenses. The Qwen and MiniMax model weights have separate terms described above.
