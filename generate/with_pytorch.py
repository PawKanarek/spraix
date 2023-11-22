import torch
import torch_xla.core.xla_model as xm
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

from generate import common

print(f"torch version: {torch.__version__}")
device = xm.xla_device()
print(f"{device=}")


def getRefinerPipeline(base) -> StableDiffusionXLPipeline:
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    refiner.to(device)
    refiner.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
    return refiner


def getBasePipeline() -> StableDiffusionXLPipeline:
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    base.to(device)
    base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
    return base


def run(
    prompt: str = common.PROMPT,
    negative_prompt: str = common.NEGATIVE_PROMPT,
    seed: int = common.SEED,
    guidance_scale: float = common.GUIDANCE_SCALE,
    steps: int = common.STEPS,
    refiner_kick_in: float = common.REFINER_KICK_IN,
):
    print("loading sdxl base...")
    base = getBasePipeline()
    images = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=steps,
        denoising_end=refiner_kick_in,
        guidance_scale=guidance_scale,
        output_type="latent",
    ).images

    print("loading sdxl refiner...")
    refiner = getRefinerPipeline(base)

    images = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        denoising_start=refiner_kick_in,
        image=images,
    ).images
    images[0].save(common.getSavePath(sub_dir="sdxl_refiner"))
