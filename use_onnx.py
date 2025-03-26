from optimum.onnxruntime import ORTStableDiffusion3Pipeline, ORTDiffusionPipeline
import onnxruntime as rt
print(rt.get_available_providers())

# The following is not yet implemented:
from optimum.onnxruntime import ORTStableDiffusion3Pipeline



model_id = "./stable-diffusion-3.5-medium-turbo"
pipeline = ORTStableDiffusion3Pipeline.from_pretrained(model_id, provider="CUDAExecutionProvider")

prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(   "A beautiful bald girl with silver and white futuristic metal face jewelry, her full body made of intricately carved liquid glass in the style of Tadashi, the complexity master of cyberpunk, in the style of James Jean and Peter Mohrbacher. This concept design is trending on Artstation, with sharp focus, studio-quality photography, and highly detailed, intricate details.",
   num_inference_steps=8,
   guidance_scale=1.5,
   height=1024,
   width=768 ).images[0]
image.save("./test4-2.webp")

