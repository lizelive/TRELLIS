import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers import GGUFQuantizationConfig

quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

transformer = SD3Transformer2DModel.from_single_file(
    "https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo/blob/main/sd3.5m_turbo-Q4_K_M.gguf",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "tensorart/stable-diffusion-3.5-medium-turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

pipe.to(device="cuda")
image = pipe("a 3d model of an Etheral character from the game SS13, with blue skin and glowing eyes. They should also holding a staff with a spiral design on top.").images[0]
image.save("sd35.png")
