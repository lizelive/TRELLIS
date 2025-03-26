import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers import SD3ControlNetModel
from diffusers.utils import load_image

controlnet = SD3ControlNetModel.from_pretrained("tensorart/SD3.5M-Controlnet-Depth", torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "tensorart/stable-diffusion-3.5-medium-turbo",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

pipe.enable_model_cpu_offload()

control_image = load_image("assets/example_image/depth_control.png")
prompt = "a 3d model of an Etheral character from the game SS13, with blue skin and glowing eyes. They should also holding a staff with a spiral design on top."
negative_prompt = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, monochrome"

image = pipe(
    prompt, 
    num_inference_steps=8,
    negative_prompt=negative_prompt, 
    control_image=control_image, 
    guidance_scale=1.5
).images[0]
image.save(f"{__file__}.webp")
