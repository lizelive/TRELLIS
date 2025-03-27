from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
import torch

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, AutoencoderTiny
from diffusers import TorchAoConfig as DiffusersTorchAoConfig

from transformers import CLIPTextModel, CLIPTextModelWithProjection

# quant_type = "float8_static_activation_float8_weight" # doesn't work becuse needs scale
# quantization_config = TorchAoConfig(quant_type, scale=torch.as_tensor(1.0))

# from torchao.quantization import int8_dynamic_activation_int4_weight
# from torchao.dtypes import CutlassInt4PackedLayout as QuantLayout
# torchao_quantization_config = int8_dynamic_activation_int4_weight(group_size=-1, layout=QuantLayout())

# quant_type = "int8_weight_only"
# quant_type = "int4dq"
quant_type = "int8_weight_only"
# quant_type = None
diffusers_quantization_config = DiffusersTorchAoConfig(
    quant_type, group_size=-1) if quant_type is not None else None

#

# TransformersTorchAoConfig("int8_dynamic_activation_int8_weight") if quant_type is not None else None
transformers_quantization_config = None

#

torch_dtype = torch.bfloat16

input_variant = None
# depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
# feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant=input_variant,
    use_safetensors=True,
    torch_dtype=torch_dtype,
    quantization_config=diffusers_quantization_config,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype, quantization_config=diffusers_quantization_config)
# vae = AutoencoderTiny.from_pretrained(
#     "madebyollin/taesdxl", torch_dtype=torch_dtype, quantization_config=diffusers_quantization_config)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
# Use the correct ckpt for your step setting!

unet = UNet2DConditionModel.from_pretrained(
    "lizelive/sdxl-lightning-unet", variant="4step", torch_dtype=torch_dtype, quantization_config=diffusers_quantization_config)


text_encoder = CLIPTextModel.from_pretrained(
    base, subfolder="text_encoder", variant=input_variant, torch_dtype=torch_dtype, quantization_config=transformers_quantization_config)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    base, subfolder="text_encoder_2", variant=input_variant, torch_dtype=torch_dtype, quantization_config=transformers_quantization_config)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base, unet=unet, vae=vae, torch_dtype=torch_dtype, text_encoder=text_encoder, text_encoder_2=text_encoder_2,  variant=input_variant, controlnet=controlnet)


# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")

pipe.save_pretrained(
    "./model/sdxl-lightning-depth-controlnet", safe_serialization=False)
