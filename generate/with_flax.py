import time

import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxStableDiffusionXLPipeline
from flax.jax_utils import replicate

from generate import common


def tokenize_prompt(pipeline, prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    NUM_DEVICES = jax.device_count()
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng


def generate_jax(
    pipeline,
    p_params,
    prompt: str = common.PROMPT,
    negative_prompt: str = common.NEGATIVE_PROMPT,
    seed: int = common.SEED,
    guidance_scale: float = common.GUIDANCE_SCALE,
    steps: int = common.STEPS,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(pipeline, prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    images = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        jit=True,
    ).images

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))


def run():
    print(f"loading sdxl base...")
    startup_time = time.time()
    pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", split_head_dim=True
    )
    scheduler_state = params.pop("scheduler")
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    params["scheduler"] = scheduler_state

    # Model parameters don't change during inference,
    # so we only need to replicate them once.
    p_params = replicate(params)

    # compile jax
    print(f"Compiling ...")
    generate_jax(pipeline, p_params, "compiling", "compiling")
    print(f"Compiled in time: {time.time() - startup_time}")

    index = 0
    for p in [common.PROMPT]:
        step_time = time.time()
        images = generate_jax(pipeline, p_params, p)
        for i in images:
            index += 1
            i.save(common.getSavePath(p, index, "sdxl"))
        print(f"Batch execution time: {time.time() - step_time}")
