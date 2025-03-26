import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image

controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-depth", torch_dtype=torch.bfloat16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()

control_image = load_image("assets/example_image/depth_control_inverted.png")
# control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth.png")

generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(
    prompt="The character in the image exudes an otherworldly and mystical presence. It has a glowing blue hue and a smooth, featureless face with piercing, luminous eyes. From its head extend intricate, horn-like structures that add a unique and elegant flair. The character is draped in flowing, layered robes that emit a soft blue glow, enhancing its ethereal aesthetic.",
    control_image=control_image,
    guidance_scale=4.5,
    num_inference_steps=40,
    generator=generator,
    max_sequence_length=77,
).images[0]
image.save(f"{__file__}.webp")
