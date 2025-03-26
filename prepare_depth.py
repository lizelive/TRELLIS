# install image_gen_aux with: pip install git+https://github.com/huggingface/image_gen_aux.git
from image_gen_aux import DepthPreprocessor
from diffusers.utils import load_image

image = load_image("assets/example_image/hair.jpg")

depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to("cuda")


depth_preprocessor(image)[0].convert("RGB").save("assets/example_image/depth_control.png")
depth_preprocessor(image, invert=True)[0].convert("RGB").save("assets/example_image/depth_control_inverted.png")
