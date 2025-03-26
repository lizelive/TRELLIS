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

from torchao.quantization import Int8DynActInt4WeightLinear
# quant_type = Float8StaticActivationFloat8WeightConfig(scale=1.0)

quant_type = "int4dq"
# quant_type = None
quantization_config = TorchAoConfig(
    quant_type) if quant_type is not None else None
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
    base, unet=unet, vae=vae, torch_dtype=torch_dtype, text_encoder=text_encoder, text_encoder_2=text_encoder_2,  variant="fp16", controlnet=controlnet).to("cuda")


# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")

pipe.save_pretrained("./model/sdxl-lightning-depth-controlnet",
                     variant=quant_type, safe_serialization=False)
