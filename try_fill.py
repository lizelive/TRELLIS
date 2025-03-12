import torch
from diffusers import FluxInpaintPipeline, FluxFillPipeline
from diffusers.utils import load_image

# model_id = "ostris/Flex.1-alpha"
# model_id = "black-forest-labs/FLUX.1-schnell"
# model_id = "black-forest-labs/FLUX.1-dev"
model_id = "black-forest-labs/FLUX.1-Fill-dev"

# pipe: FluxInpaintPipeline = FluxInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

# pipe.to("cuda")
# prompt = "an empty park bench"
# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
# image = load_image(img_url)
# mask = load_image(mask_url)
# image = pipe(prompt=prompt, image=image, mask_image=mask,
#              width=512, height=512,
#              guidance_scale=0.0, num_inference_steps=4, strength=0.8,
#              max_sequence_length=256).images[0]
# image.save("flux_inpainting.png")

image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux_fill.png")