from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
import torch

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, AutoencoderTiny
from diffusers import TorchAoConfig as DiffusersTorchAoConfig

from transformers import CLIPTextModel, CLIPTextModelWithProjection


import torch.ao.quantization as aoq
torch.quint4x2

def quantize(model):
    # return aoq.quantize_dynamic(model, dtype=torch.quint4x2)
    return aoq.quantize_dynamic(model, dtype=torch.qint8)

transformers_quantization_config = None


torch_dtype = torch.float32

input_variant = "fp16"

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant=input_variant,
    use_safetensors=True,
    torch_dtype=torch_dtype,
)

controlnet = quantize(controlnet)


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)

# vae = AutoencoderTiny.from_pretrained(
#     "madebyollin/taesdxl", torch_dtype=torch_dtype, quantization_config=diffusers_quantization_config)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
# Use the correct ckpt for your step setting!

unet = UNet2DConditionModel.from_pretrained(
    "lizelive/sdxl-lightning-unet", variant="4step", torch_dtype=torch_dtype)

text_encoder = CLIPTextModel.from_pretrained(
    base, subfolder="text_encoder", variant=input_variant, torch_dtype=torch_dtype, quantization_config=transformers_quantization_config)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    base, subfolder="text_encoder_2", variant=input_variant, torch_dtype=torch_dtype, quantization_config=transformers_quantization_config)





pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base, unet=quantize(unet), vae=quantize(vae), torch_dtype=torch_dtype, text_encoder=quantize(text_encoder), text_encoder_2=quantize(text_encoder_2),  variant=input_variant, controlnet=controlnet)


# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")

# pipe.save_pretrained(
#     "./model/sdxl-lightning-depth-controlnet", safe_serialization=False)

# pipe = pipe.to("cuda")

prompt = "a 3d model of an Etheral character from the game SS13, with blue skin and glowing eyes. They should also holding a staff with a spiral design on top."
controlnet_conditioning_scale = 0.5  # recommended for good generalization


from diffusers.utils import load_image
from with_timer import Timer


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
