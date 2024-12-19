import os
from datetime import datetime
from diffusers import DiffusionPipeline
import argparse

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

# run pipeline in inference (sample random noise and denoise)

# Parse the prompt argument
parser = argparse.ArgumentParser(description="Take a string prompt as an argument.")
parser.add_argument("prompt", type=str, help="The text prompt to use")
args = parser.parse_args()

# Now you can use the prompt
prompt = args.prompt
print(f"Prompt received: {prompt}")
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

output_dir = "results/compVis"
os.makedirs(output_dir, exist_ok=True)

# Save images with timestamped filenames
for idx, image in enumerate(images):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(os.path.join(output_dir, f"squirrel-{timestamp}_{idx}.png"))
