import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from generate import common

from diffusers import FlaxStableDiffusionPipeline


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
    print(f"loading sd v1-4 finetuned model...")
    startup_time = time.time()
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        "/mnt/disks/persist/repos/diffusers/examples/text_to_image/spraix_sd_1_4_flax",
        # split_head_dim=True,
        dtype=jnp.bfloat16,
        safety_checker=None,
    )
    # scheduler_state = params.pop("scheduler")
    # params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    # params["scheduler"] = scheduler_state

    # Model parameters don't change during inference,
    # so we only need to replicate them once.
    p_params = replicate(params)

    # compile jax
    print(f"Compiling ...")
    generate_jax(pipeline, p_params, "compiling", "compiling")
    print(f"Compiled in time: {time.time() - startup_time}")

    prompts = [
        "12-frame sprite animation of: cute small dinosaur with backpack, that: is running, facing: East",
        "4-frame sprite animation of: girl with big sword, that: is idle, facing: West",
        "8-frame sprite animation of: a horned devil with big lasers, that: is jumping, facing: West",
        "6-frame sprite animation of: a slime, that: is fainting, facing: East",
    ]
    index = 0
    for i, prompt in enumerate(prompts):
        step_time = time.time()
        images = generate_jax(pipeline, p_params, prompt)
        for i in images:
            index += 1
            i.save(common.getSavePath(prompt, index, "spraix_sd_1_4_flax"))
        print(f"Batch execution time: {time.time() - step_time}")
