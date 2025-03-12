import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig
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

dtype = torch.bfloat16

# from optimum.quanto import freeze, qfloat8, quantize

quant_type = "int8_weight_only" # "int8_weight_only" # "float8wo" # "int8wo" # "float8wo_e5m2" # "float8wo_e4m3"
quantization_config = TorchAoConfig(quant_type)
transformer : FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(
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

transformer = transformer.to(memory_format=torch.channels_last)

# transformer.fuse_qkv_projections()
# pipe.vae.fuse_qkv_projections()
#
# "max-autotune"
# transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)

pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    text_encoder_2 = text_encoder_2,
    torch_dtype=dtype,
)

# pipe.set_progress_bar_config(disable=True)

# pipe
# pipe.vae.to(memory_format=torch.channels_last)

print(torch.cuda.memory_summary())
pipe.to("cuda")
# print(torch.cuda.memory_summary())
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()


# pipe.enable_model_cpu_offload()
# Without quantization: ~31.447 GB
# With quantization: ~20.40 GB
print(torch.cuda.memory_summary())
print(
    f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt, num_inference_steps=4, guidance_scale=0., max_sequence_length=256
).images[0]
print(torch.cuda.memory_summary())

image = pipe(
    prompt, num_inference_steps=4, guidance_scale=0., max_sequence_length=256
).images[0]

image.save(f"output_{quant_type}.png")
print(torch.cuda.memory_summary())
