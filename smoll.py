from diffusers import ModelMixin
from typing import Type, TypeVar, cast, assert_type
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig, AutoencoderKL
from transformers import T5EncoderModel
# from diffusers import QuantoConfig


torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# model_id = "black-forest-labs/FLUX.1-dev"
# model_id = "ostris/Flex.1-alpha"
model_id = "black-forest-labs/FLUX.1-schnell"

torch_dtype = torch.bfloat16

# from optimum.quanto import freeze, qfloat8, quantize

# "int8_weight_only" # "float8wo" # "int8wo" # "float8wo_e5m2" # "float8wo_e4m3"
quant_type = "int8_weight_only"
quantization_config = TorchAoConfig(quant_type)


ModelType = TypeVar("ModelType")


def load_submodel(kind: Type[ModelType], model_id: str, subfolder: str, quantization_config: TorchAoConfig) -> ModelType:
    model = kind.from_pretrained(
        pretrained_model_name_or_path=model_id,
        subfolder=subfolder,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    if subfolder in ("transformer", "vae"):
        model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    # assert_type(model, kind)
    return model

def load_model() -> FluxPipeline:
    transformer = load_submodel(
        kind=FluxTransformer2DModel,
        model_id=model_id,
        subfolder="transformer",
        quantization_config=quantization_config
    )

    text_encoder_2 = load_submodel(
        kind=T5EncoderModel,
        model_id=model_id,
        subfolder="text_encoder_2",
        quantization_config=quantization_config
    )

    vae = load_submodel(
        kind=AutoencoderKL,
        model_id=model_id,
        subfolder="vae",
        quantization_config=quantization_config
    )

    # transformer.fuse_qkv_projections()
    # pipe.vae.fuse_qkv_projections()
    #
    # "max-autotune"
    # transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)

    # scheduler: FlowMatchEulerDiscreteScheduler,
    # vae: AutoencoderKL,
    # text_encoder: CLIPTextModel,
    # tokenizer: CLIPTokenizer,
    # text_encoder_2: T5EncoderModel,
    # tokenizer_2: T5TokenizerFast,
    # transformer: FluxTransformer2DModel,

    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    )
    # pipe.set_progress_bar_config(disable=True)
    return pipe



# pipe
# pipe.vae.to(memory_format=torch.channels_last)

pipe = load_model()

pipe.to("cuda")
# print(torch.cuda.memory_summary())
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()


# pipe.enable_model_cpu_offload()
# Without quantization: ~31.447 GB
# With quantization: ~20.40 GB
print(
    f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

import time
for j in range(8):
    prompt = "a Full body 3d model of an ancient tree person, made of roots interwoven with blue glowing crystals. a white background. she looks lonely."
    start = time.time()

    images = pipe(
        prompt, num_inference_steps=4, guidance_scale=0., max_sequence_length=256, num_images_per_prompt=2
    ).images
    end = time.time()
    duration = end - start
    print(duration)

    for i, image in enumerate(images):
        image.save(f"output_{quant_type}_{j}_{i}.png")
print(torch.cuda.memory_summary())
