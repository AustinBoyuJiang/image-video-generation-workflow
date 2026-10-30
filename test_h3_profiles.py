"""Validate H3 sampler graphs without loading model weights or writing media."""
import io

from PIL import Image

from h3_runner import MODEL_PROFILES, _graph, profile_steps


buffer = io.BytesIO()
Image.new("RGB", (640, 360), "navy").save(buffer, format="PNG")
source = buffer.getvalue()

pika, _, _, _ = _graph("Camera pushes in.", source, 7, "test-token", "pika")
assert profile_steps("pika") == 6
assert pika["7"]["inputs"]["lora_name"] == "pika_h3_flex_fl2va_v1_bf16.safetensors"
assert pika["10"]["inputs"]["sampler_name"] == "lcm"
assert pika["11"] == {
    "class_type": "ManualSigmas",
    "inputs": {"sigmas": "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"},
}

turbo, _, _, _ = _graph("Camera pushes in.", source, 7, "test-token", "fast")
assert profile_steps("fast") == 4
assert turbo["7"]["inputs"]["lora_name"].startswith("minimax_h3_fl2v_turbo_4step")
assert turbo["10"]["inputs"]["sampler_name"] == "res_multistep"
assert turbo["11"]["class_type"] == "BasicScheduler"

base, _, _, _ = _graph("Camera pushes in.", source, 7, "test-token", "base")
assert profile_steps("base") == 50
assert "7" not in base
assert base["11"]["inputs"]["steps"] == 50
assert set(MODEL_PROFILES) == {"fast", "base", "pika"}

print("PASS: Turbo, Base, and Pika H3 Flex use their exact LoRA, sampler, and sigma profiles")
