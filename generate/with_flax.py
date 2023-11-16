import time

import jax
import jax.numpy as jnp
import numpy as np
from diffusers import (
    FlaxStableDiffusionXLImg2ImgPipeline,
    FlaxStableDiffusionXLPipeline,
)
from flax.jax_utils import replicate

from generate import common


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    NUM_DEVICES = jax.device_count()
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng


def generate_jax(
    pipeline,
    pipeline_r,
    p_params,
    p_params_r,
    prompt: str = common.PROMPT,
    negative_prompt: str = common.NEGATIVE_PROMPT,
    seed: int = common.SEED,
    guidance_scale: float = common.GUIDANCE_SCALE,
    steps: int = common.STEPS,
):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)

    base_latents = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        # denoising_end=common.REFINER_KICK_IN,
        output_type="latent",
        jit=True,
    ).images

    images = pipeline_r(
        prompt_ids=prompt_ids,
        params=p_params_r,
        prng_seed=rng,
        num_inference_steps=steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        # denoising_start=common.REFINER_KICK_IN,
        image=base_latents,
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

    print("loading refiner...")
    pipeline_r, params_r = FlaxStableDiffusionXLImg2ImgPipeline.from_pretrained(
        "pcuenq/stable-diffusion-xl-refiner-1.0-flax", split_head_dim=True
    )
    scheduler_state_r = params_r.pop("scheduler")
    del params_r["vae"]
    del params_r["text_encoder_2"]
    params_r = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params_r)
    params_r["vae"] = params["vae"]
    params_r["text_encoder_2"] = params["text_encoder_2"]
    params_r["scheduler"] = scheduler_state_r

    # Model parameters don't change during inference,
    # so we only need to replicate them once.
    p_params = replicate(params)
    p_params_r = replicate(params_r)

    # compile jax
    print(f"Compiling ...")
    generate_jax(pipeline, pipeline_r, p_params, p_params_r)
    print(f"Compiled in time: {time.time() - startup_time}")

    index = 0
    for i in range(1):
        step_time = time.time()
        images = generate_jax(pipeline, pipeline_r, p_params, p_params_r)
        for i in images:
            index += 1
            i.save(common.getSavePath(index))
        print(f"Batch execution time: {time.time() - step_time}")
