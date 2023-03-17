import os
from os import path
import re
import sys

import torch

from diffusers import StableDiffusionPipeline

diffusers_dir = path.join(path.dirname(__file__), "..", "..", "diffusers")
script_file = path.join(diffusers_dir, "scripts", "convert_diffusers_to_original_stable_diffusion.py")

model_id = sys.argv[1]

output = re.sub(r'[^A-Za-z0-9]+', '-', model_id) + ".safetensors"

model, cached_dir = StableDiffusionPipeline.from_pretrained(
    sys.argv[1],
    torch_dtype=torch.float16,
    resume_download=True,
    return_cached_folder=True,
)

os.system(
    f'python "{script_file}" '
    f'--model_path "{cached_dir}" '
    f'--checkpoint_path "{output}" '
    f'--use_safetensors')
