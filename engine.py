"""Official Qwen Image 2.1 generation and instruction editing, in memory."""
import io,os,secrets,threading,time
from pathlib import Path
import torch
from PIL import Image,ImageOps
from diffusers import QwenImage21Pipeline
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"qwen-image-2.1"
LOCK=threading.Lock()
PIPE=None
Image.MAX_IMAGE_PIXELS=12_000_000

def load_pipe(notify):
    global PIPE
    if PIPE is not None:return PIPE
    notify("loading")
    runtime=os.environ.get("QWEN_RUNTIME","bf16_group")
    device=torch.device("cuda",int(os.environ.get("QWEN_GPU",os.environ.get("Z_IMAGE_GPU","1"))))
    pipe=QwenImage21Pipeline.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    if runtime=="bf16_group":
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(pipe.text_encoder,onload_device=device,offload_device=torch.device("cpu"),
            offload_type="leaf_level",use_stream=True,non_blocking=True,record_stream=True)
        pipe.transformer.enable_group_offload(onload_device=device,offload_device=torch.device("cpu"),
            offload_type="block_level",num_blocks_per_group=1,use_stream=True,non_blocking=True,record_stream=True)
        pipe.vae.to(device)
    elif runtime=="int8_offload":
        from optimum.quanto import quantize,freeze,qint8
        for component in (pipe.text_encoder,pipe.transformer):
            quantize(component,weights=qint8)
            freeze(component)
        pipe.enable_model_cpu_offload(gpu_id=device.index)
    elif runtime=="bf16_sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=device.index)
    else:raise ValueError("Unsupported runtime")
    PIPE=pipe
    return PIPE

def generate(prompt,width,height,seed,steps,image_bytes=None,strength=None,progress=None):
    with LOCK:
        total_start=time.monotonic()
        notify=progress or (lambda *args:None)
        device=torch.device("cuda",int(os.environ.get("QWEN_GPU",os.environ.get("Z_IMAGE_GPU","1"))))
        torch.cuda.set_device(device);torch.cuda.init()
        started_load=time.monotonic()
        pipe=load_pipe(notify)
        load_seconds=time.monotonic()-started_load
        notify("preparing")
        source=None
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as original:
                if original.width*original.height>Image.MAX_IMAGE_PIXELS:raise ValueError("Reference image too large")
                source=ImageOps.exif_transpose(original).convert("RGB")
                source.thumbnail((1536,1536),Image.Resampling.LANCZOS)
        if seed<0:seed=secrets.randbelow(2**32)
        def on_step(p,step,timestep,kwargs):
            notify("sampling",step+1,steps)
            return kwargs
        options=dict(prompt=prompt,width=width,height=height,output_resolution=max(width,height),
            num_inference_steps=steps,true_cfg_scale=1.0,use_kv_cache=True,
            generator=torch.Generator("cpu").manual_seed(seed),callback_on_step_end=on_step)
        if source is not None:options["image"]=source
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        started=time.monotonic();notify("encoding")
        try:
            with torch.inference_mode():
                result=pipe(**options).images[0]
            torch.cuda.synchronize(device)
            seconds=time.monotonic()-started
            notify("decoding")
            buffer=io.BytesIO()
            result.info.clear();result.save(buffer,format="PNG")
            result.close();payload=buffer.getvalue();buffer.close()
            return payload,dict(seed=seed,seconds=round(seconds,2),load_seconds=round(load_seconds,2),
                total_seconds=round(time.monotonic()-total_start,2),
                peak_vram_gib=round(torch.cuda.max_memory_allocated(device)/2**30,2),
                model="Qwen/Qwen-Image-2.1",runtime=os.environ.get("QWEN_RUNTIME","bf16_group"))
        finally:
            if source is not None:source.close()
            options.clear()

def release():
    global PIPE
    if PIPE is not None:
        try:PIPE.remove_all_hooks()
        except Exception:pass
    PIPE=None
    import gc
    gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()
