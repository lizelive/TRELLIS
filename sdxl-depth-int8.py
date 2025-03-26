import torch
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from diffusers import TorchAoConfig

quant_type = "int8_weight_only"
quantization_config = TorchAoConfig(quant_type)
torch_dtype = torch.bfloat16

# depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
# feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    use_safetensors=True,
    torch_dtype=torch_dtype,
    quantization_config=quantization_config,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", quantization_config=quantization_config, torch_dtype=torch_dtype)

unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="unet",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    unet=unet,
    use_safetensors=True,
    torch_dtype=torch_dtype,
    # quantization_config=quantization_config,
)

pipe.to("cuda")


# def get_depth_map(image):
#     image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
#     with torch.no_grad(), torch.autocast("cuda"):
#         depth_map = depth_estimator(image).predicted_depth

#     # depth_map=depth_map.unsqueeze(1)
#     depth_map = torch.nn.functional.interpolate(
#         depth_map.unsqueeze(1),
#         size=(1024, 1024),
#         mode="bicubic",
#         align_corners=False,
#     )
#     depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#     image = torch.cat([depth_map] * 3, dim=1)

#     image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
#     image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
#     return image


prompt = "a 3d model of an Etheral character from the game SS13, with blue skin and glowing eyes. They should also holding a staff with a spiral design on top."
# image = load_image("https://media.discordapp.net/attachments/1294558841861570561/1345696211281248277/OIG4.png?ex=67d9ec02&is=67d89a82&hm=805f1177ecf36e06ceec4cfca26b04b255899e8823302b156625a7c2e0dc19d8&=&format=webp&quality=lossless")
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = load_image("assets/example_image/depth_control.png")

images = pipe(
    prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,
).images

images[0].save(f"{__file__}.webp")
