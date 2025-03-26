from torchao.quantization import Float8StaticActivationFloat8WeightConfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
import torch
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from diffusers import TorchAoConfig

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# quant_type = "float8_static_activation_float8_weight" # doesn't work becuse needs scale
# quantization_config = TorchAoConfig(quant_type, scale=torch.as_tensor(1.0))


# quant_type = Float8StaticActivationFloat8WeightConfig(scale=1.0)

quant_type = "int4dq"
quantization_config = TorchAoConfig(quant_type)
torch_dtype = torch.float16

# depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
# feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch_dtype,
    quantization_config=quantization_config,
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype, quantization_config=quantization_config)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
# Use the correct ckpt for your step setting!

unet = UNet2DConditionModel.from_pretrained(
    "lizelive/sdxl-lightning-unet", variant="4step", torch_dtype=torch_dtype, quantization_config=quantization_config)


text_encoder = CLIPTextModel.from_pretrained(
    base, subfolder="text_encoder", variant="fp16", torch_dtype=torch_dtype, quantization_config=quantization_config)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    base, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch_dtype, quantization_config=quantization_config)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base, unet=unet, torch_dtype=torch_dtype, text_encoder=text_encoder, text_encoder_2=text_encoder_2,  variant="fp16", controlnet=controlnet).to("cuda")


# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")

pipe.save_pretrained("./model/sdxl-lightning-depth-controlnet",
                     variant="int4dq", safe_serialization=False)

prompt = "a 3d model of an Etheral character from the game SS13, with blue skin and glowing eyes. They should also holding a staff with a spiral design on top."
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = load_image("assets/example_image/depth_control.png")

images = pipe(
    prompt, image=depth_image, num_inference_steps=4, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=0
).images
images[0].save(f"{__file__}.{quant_type}.webp")
