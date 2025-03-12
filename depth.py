import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
from diffusers import FluxTransformer2DModel, TorchAoConfig
from transformers import DepthAnythingForDepthEstimation, AutoImageProcessor, T5EncoderModel
import accelerate
from transformers import pipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path

from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(3)
])

dtype = torch.bfloat16

quant_type = "int8_weight_only"
quantization_config = TorchAoConfig(quant_type)

depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", model_kwargs={
    "quantization_config": quantization_config
}, use_fast=True)


model_id = "black-forest-labs/FLUX.1-Depth-dev"

save_dir = Path("/tmp")/model_id/quant_type


def preprocess_depth_control_image(control_image):
    est = depth_pipe(control_image)
    # convert("RGB")#
    predicted_depth = est["depth"]
    # pdt = pil_to_tensor(predicted_depth)
    # print("max", pdt.aminmax())
    return predicted_depth.convert("RGB")

def load_from_hf(model_id) -> FluxControlPipeline:
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )

    # transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)

    pipe = FluxControlPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
    )
    return pipe

# pipe = load_from_hf(model_id)
# pipe.save_pretrained(save_dir, safe_serialization=False)
pipe = FluxControlPipeline.from_pretrained(save_dir, torch_dtype=dtype, use_safetensors=False)
# pipe.enable_sequential_cpu_offload()
pipe.enable_model_cpu_offload()

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

control_image = preprocess_depth_control_image(control_image)

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output.png")
