import time
import os

import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxStableDiffusionXLPipeline
from flax.jax_utils import replicate

from generate import common

MODEL = "spraix_sdxl_9frames_start_6epoch"
END = 7  # Set 0 to generate only final model, otherwise set number of epochs to check every checkpoint
START = 5


def tokenize_prompt(pipeline, prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, jax.device_count())
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
    for i in range(START, END + 1):
        model = MODEL
        if i == 0:
            continue  # for 32 epochs i didn't create main model; so skip it

        if i > 0:
            model = f"{model}_epoch_{i}"

        print(f"loading {model}")
        startup_time = time.time()
        pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
            f"/home/raix/w/spraix/{model}", split_head_dim=True
        )
        scheduler_state = params.pop("scheduler")
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
        params["scheduler"] = scheduler_state

        # Model parameters don't change during inference,
        # so we only need to replicate them once.
        p_params = replicate(params)

        # compile jax
        print("Compiling ...")
        generate_jax(pipeline, p_params, "compiling", "compiling")
        print(f"Compiled in time: {time.time() - startup_time}")

        prompts = [
            "Pixel-art animation of a blue water droplet with legs, that: is swiniging axe, facing: East",
            "Pixel-art animation of a Tree, that: is Idle, facing: South",
            "Pixel-art animation of a Dinosaur with a backpack, that: is jumping, facing: North",
            "Pixel-art animation of a Fire demon with axe, that: is running, facing West",
        ]
        index = 0
        for prompt in prompts:
            step_time = time.time()
            images = generate_jax(pipeline, p_params, prompt)
            for img in images:
                index += 1
                pth = os.path.join(MODEL, f"_epoch_{i}")
                img.save(common.getSavePath(prompt, index, pth))
            print(f"Batch execution time: {time.time() - step_time}")
