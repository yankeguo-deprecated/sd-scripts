import sys

import torch

from diffusers import StableDiffusionPipeline

model, cached_dir = StableDiffusionPipeline.from_pretrained(
    sys.argv[1],
    torch_dtype=torch.float16,
    resume_download=True,
    return_cached_folder=True,
)

print("cached to: " + cached_dir)
