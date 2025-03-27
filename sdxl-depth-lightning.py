import torch


import torchao.dtypes
torch.serialization.add_safe_globals([torchao.dtypes.MarlinQQQTensor])

torch.set_float32_matmul_precision("medium")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
# https://pytorch.org/blog/accelerating-generative-ai-3/

torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True


# torch._inductor.config.compile_threads = 16

from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from with_timer import Timer



torch_dtype = torch.bfloat16

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "./model/sdxl-lightning-depth-controlnet", torch_dtype=torch_dtype, use_safetensors=False).to("cuda:0")

# disable the progress bar
pipe.set_progress_bar_config(disable=True)  

# pipe.fuse_qkv_projections()

# compile_mode = "max-autotune"
pipe.unet = torch.compile(pipe.unet, mode="default", fullgraph=True, dynamic=False)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="default", fullgraph=True, dynamic=False)





prompt = "a 3d model of an Etheral character from the game SS13, with blue skin and glowing eyes. They should also holding a staff with a spiral design on top."
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = load_image("assets/example_image/depth_control.png")

# print torch memory summary


# start time
for i in range(4):
    with Timer("inference"):
        images = pipe(
            prompt, image=depth_image, num_inference_steps=4, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=0
        ).images
    images[0].save(f"{__file__}.{i}.webp")

print(torch.cuda.memory_summary())
