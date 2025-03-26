import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download





base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
# Use the correct ckpt for your step setting!
ckpt = "sdxl_lightning_4step_unet.safetensors"

name = "lizelive/sdxl-lightning-unet"

# Load model.
# from torchao.float8 import convert_to_float8_training
unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(base, subfolder="unet")).to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
unet.save_pretrained(name, variant="4step", push_to_hub=True)
