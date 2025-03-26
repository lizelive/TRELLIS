import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers import SD3ControlNetModel
from diffusers.utils import load_image

controlnet = SD3ControlNetModel.from_pretrained("tensorart/SD3.5M-Controlnet-Depth", torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

pipe.enable_model_cpu_offload()

control_image = load_image("assets/example_image/depth_control.png")
prompt = "The character in the image exudes an otherworldly and mystical presence. It has a glowing blue hue and a smooth, featureless face with piercing, luminous eyes. From its head extend intricate, horn-like structures that add a unique and elegant flair. The character is draped in flowing, layered robes that emit a soft blue glow, enhancing its ethereal aesthetic."
negative_prompt = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, monochrome"

image = pipe(
    prompt, 
    num_inference_steps=30,
    negative_prompt=negative_prompt, 
    control_image=control_image, 
    guidance_scale=4.5,
).images[0]
image.save(f"{__file__}.webp")