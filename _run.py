import io
import os
import re
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline
import time

import torch
import torch_xla.core.xla_model as xm
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

PROMPT = "An expressionist style oil painting with vivid colors of exploding nebula of Sad Pepe the Frog meme in afro hair, with mustaches on width of nose"
NEGATIVE_PROMPT = "blurry, ugly, low-quality, deformed"
STEPS = 40
DEFAULT_SEED = 33
DEFAULT_GUIDANCE_SCALE = 5.0
REFINER_KICK_IN = 0.8
IMG_NAME = ""  # leave empty to auto generate
BASE_ID = "/mnt/disks/persist/repos/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "output/"
NUM_DEVICES = jax.device_count()


def getSavePath() -> str:
    name = IMG_NAME
    if not name:
        name = PROMPT

    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = name.replace(" ", "_")[:200]
    unique_id = str(uuid.uuid4())
    file_name = f"{name}_{unique_id}.png"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    print(f"saving to {file_path}")
    return file_path


def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng


def generate(
    prompt,
    negative_prompt,
    seed=DEFAULT_SEED,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=STEPS,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    images = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=num_inference_steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        jit=True,
    ).images

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))


if __name__ == "__main__":
    start_time = time.time()

    pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
        BASE_ID,
        split_head_dim=True,
    )
    scheduler_state = params.pop("scheduler")
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    params["scheduler"] = scheduler_state

    # Model parameters don't change during inference,
    # so we only need to replicate them once.
    p_params = replicate(params)

    print(f"Compiling ...")

    images = generate(PROMPT, NEGATIVE_PROMPT)
    for i in images:
        i.save(getSavePath())

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Compiled in time: {execution_time}")
