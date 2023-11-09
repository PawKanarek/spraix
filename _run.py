import io
import os
import re
import time
import uuid

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
REFINER_KICK_IN = 0.8
IMG_NAME = ""  # leave empty to auto generate
BASE_ID = "../stable-diffusion-xl-base-1.0"
REFINER_ID = "../stable-diffusion-xl-refiner-1.0"
VAE_ID = "../sdxl-vae-fp16-fix"
OUTPUT_DIR = "output/"
dev = xm.xla_device()


def withRefiner():
    # TODO: add images_per_promt (i have probably too little ram for this!)
    # num_images_per_prompt = 4
    # if (num_images_per_prompt == 1):
    # else:
    #     for i, image in enumerate(images.images):
    #         image.save(os.path.join(output_dir, f"{img_name}_{i}{img_extension}"))
    base = getBasePipeline()
    # print(base.scheduler) #[<class 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'>, <class 'diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler'>, <class 'diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler'>, <class 'diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler'>, <class 'diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler'>, <class 'diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler'>, <class 'diffusers.schedulers.scheduling_pndm.PNDMScheduler'>, <class 'diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler'>, <class 'diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler'>, <class 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler'>, <class 'diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler'>, <class 'diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler'>, <class 'diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler'>, <class 'diffusers.utils.dummy_torch_and_scipy_objects.LMSDiscreteScheduler'>]
    images = base(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS,
        denoising_end=REFINER_KICK_IN,
        output_type="latent",
    ).images
    refiner = getRefiner(base)
    # print(refiner.scheduler) #[<class 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'>, <class 'diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler'>, <class 'diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler'>, <class 'diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler'>, <class 'diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler'>, <class 'diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler'>, <class 'diffusers.schedulers.scheduling_pndm.PNDMScheduler'>, <class 'diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler'>, <class 'diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler'>, <class 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler'>, <class 'diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler'>, <class 'diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler'>, <class 'diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler'>, <class 'diffusers.utils.dummy_torch_and_scipy_objects.LMSDiscreteScheduler'>]
    images = refiner(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS,
        denoising_start=REFINER_KICK_IN,
        image=images,
    ).images
    images[0].save(getSavePath())


def getRefiner(base) -> StableDiffusionXLPipeline:
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_ID,
        vae=base.vae,
        text_encoder_2=base.text_encoder_2,
        use_safetensors=True,
    )
    refiner.to(device)
    refiner.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
    refiner.enable_attention_slicing()
    return refiner


def getBasePipeline() -> StableDiffusionXLPipeline:
    vae = AutoencoderKL.from_pretrained(VAE_ID)
    base = StableDiffusionXLPipeline.from_pretrained(
        BASE_ID, use_safetensors=True, vae=vae
    )
    base.to(device)
    base.enable_attention_slicing()
    base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
    return base


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


if __name__ == "__main__":
    start_time = time.time()
    withRefiner()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
