import argparse
import functools
import logging
import math
import os
import random
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTokenizer,
    FlaxCLIPTextModel,
    FlaxCLIPTextModelWithProjection,
    set_seed,
)

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionXLPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = logging.getLogger(__name__)

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=True "
#     "--xla_gpu_enable_async_collectives=true "
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_highest_priority_async_stream=true "
# )


device_mesh = mesh_utils.create_device_mesh((1, 2, 4))
print(device_mesh)
mesh = Mesh(devices=device_mesh, axis_names=("all", "split1", "split2"))
print(mesh)


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--from_pt",
        action="store_true",
        default=False,
        help="Flag to indicate whether to convert models from PyTorch.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x, params))


# We need to tokenize input captions and transform the images.
def encode_prompt(
    batch,
    text_encoder_1: FlaxCLIPTextModel,
    text_encoder_2: FlaxCLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    proportion_empty_prompts,
    caption_column,
    dtype,
    is_train=True,
):
    print("encode prompt")
    prompt_batch = batch[caption_column]
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )

    # from pipeline_flax_stable_diffusion_xl.py
    print("compute text imputs from tokenizer 1")
    text_inputs_1 = tokenizer_1(
        captions,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    prompt_embeds_1_out = text_encoder_1(
        text_inputs_1.input_ids,
        params=text_encoder_1.params,
        output_hidden_states=True,
    )
    prompt_embeds_1 = prompt_embeds_1_out["hidden_states"][-2]
    print(f"{prompt_embeds_1.shape=}, {prompt_embeds_1.dtype=}")
    text_inputs_2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    print(f"ids from tokenizer2  {text_inputs_2.input_ids.shape=}")
    prompt_embeds_2_out = text_encoder_2(
        text_inputs_2.input_ids, params=text_encoder_2.params, output_hidden_states=True
    )
    # We are only ALWAYS interested in the pooled output of the final text encoder
    prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]

    pooled_prompt_embeds = prompt_embeds_2_out["text_embeds"]
    prompt_embeds = jnp.concatenate([prompt_embeds_1, prompt_embeds_2], axis=-1)
    print(f"{prompt_embeds.shape=}, {pooled_prompt_embeds.shape=}")
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def compute_vae_encodings(batch, vae: FlaxAutoencoderKL, vae_params, dtype, seed):
    print("encode vae encodings")
    images = batch.pop("pixel_values")
    pixel_values = jnp.stack([jnp.array(image) for image in images])
    pixel_values = pixel_values.astype(dtype)

    # model_input = vae.encode(pixel_values)["latent_dist"].sample(seed=jax.random.PRNGKey(seed))

    vae_out = vae.apply(
        {"params": vae_params}, pixel_values, deterministic=True, method=vae.encode
    )
    latents = vae_out.latent_dist.sample(jax.random.PRNGKey(seed))
    # They do dis in train_controlnet_flax (NHWC) -> (NCHW)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    model_input = latents * vae.config.scaling_factor
    return {"model_input": model_input}


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    # jax_devices = jax.local_devices()
    # jax_devices = jax_devices[1:]
    # n_devices = len(jax_devices)
    # n_devices = 1
    # I commented out n_devices to fix parallelis problems (but i dont know if this is good idea)
    total_train_batch_size = args.train_batch_size  # * n_devices
    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    text_encoder_1 = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="text_encoder",
        # dtype=weight_dtype,
    )
    text_encoder_2 = FlaxCLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="text_encoder_2",
        # dtype=weight_dtype,
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="vae",
        # dtype=weight_dtype,
        # sharding=mesh_sharding(PartitionSpec("all")),
    )

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="unet",
        # dtype=weight_dtype,
        # sharding=mesh_sharding(PartitionSpec("all")),
    )

    # if weight_dtype == jnp.float16:
    #     print("converting weights to fp16")
    #     unet_params = unet.to_fp16(unet_params)
    #     vae_params = vae.to_fp16(vae_params)
    # elif weight_dtype == jnp.bfloat16:
    #     print("converting weights to bf16")
    #     unet_params = unet.bf_16(unet_params)
    #     vae_params = vae.bf_16(vae_params)

    # Preprocessing the datasets.
    train_resize = transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_crop = (
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution)
    )
    train_flip = (
        transforms.RandomHorizontalFlip()
        if args.random_flip
        else transforms.Lambda(lambda x: x)
    )
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(
                    image, (args.resolution, args.resolution)
                )
                image = crop(image, y1, x1, h, w)
            if args.random_flip and random.random() < 0.5:
                # flip
                x1 = image.width - x1
                image = train_flip(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        print("finish preproces")
        return examples

    if args.max_train_samples is not None:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=args.seed)
            .select(range(args.max_train_samples))
        )
        # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.
    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        proportion_empty_prompts=args.proportion_empty_prompts,
        caption_column=args.caption_column,
        dtype=weight_dtype,
    )
    compute_vae_encodings_fn = functools.partial(
        compute_vae_encodings,
        vae=vae,
        vae_params=vae_params,
        dtype=weight_dtype,
        seed=args.seed,
    )

    if jax.process_index() == 0:
        from datasets.fingerprint import Hasher

        print("compute all embedings")

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        new_fingerprint_for_vae = Hasher.hash("vae")
        print("compute all embedings 1")

        train_dataset = train_dataset.map(
            compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
        )
        print("compute all embedings 2")
        train_dataset = train_dataset.map(
            compute_vae_encodings_fn,
            batched=True,
            batch_size=args.train_batch_size,  # * n_devices,  dont use parallelism for now
            new_fingerprint=new_fingerprint_for_vae,
        )

    def collate_fn(examples):
        model_input = jnp.stack(
            [jnp.asarray(example["model_input"]) for example in examples]
        )
        original_sizes = jnp.asarray(
            [example["original_sizes"] for example in examples]
        )
        crop_top_lefts = jnp.asarray(
            [example["crop_top_lefts"] for example in examples]
        )
        prompt_embeds = jnp.stack(
            [jnp.asarray(example["prompt_embeds"]) for example in examples]
        )
        pooled_prompt_embeds = jnp.stack(
            [jnp.asarray(example["pooled_prompt_embeds"]) for example in examples]
        )

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=total_train_batch_size,
        drop_last=True,
    )

    if weight_dtype == jnp.float16:
        unet_params = unet.to_fp16(unet_params)
        vae_params = vae.to_fp16(vae_params)
    elif weight_dtype == jnp.bfloat16:
        print("converting weights to bf16")
        unet_params = unet.to_bf16(unet_params)
        vae_params = vae.to_bf16(vae_params)

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    state = train_state.TrainState.create(
        apply_fn=unet.__call__, params=unet_params, tx=optimizer
    )

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    rng = jax.random.PRNGKey(args.seed)
    # train_rngs = jax.random.split(rng, n_devices)
    train_rngs = rng

    def train_step(state, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Sample noise that we'll add to the latents
            model_input = batch["model_input"]
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, model_input.shape)
            bsz = model_input.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )
            timesteps = timesteps

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(
                noise_scheduler_state, model_input, noise, timesteps
            )

            # time ids
            def compute_time_ids(original_size, crops_coords_top_left):
                # original_size = jnp.squeeze(original_size)
                # crops_coords_top_left = jnp.squeeze(crops_coords_top_left)
                target_size = jnp.asarray((args.resolution, args.resolution))
                add_time_ids = jnp.concatenate(
                    [original_size, crops_coords_top_left, target_size]
                )
                print(
                    f"{add_time_ids.shape=}=cat({original_size.shape=},{crops_coords_top_left.shape=},{target_size.shape=})"
                )
                add_time_ids = jnp.array([add_time_ids], dtype=weight_dtype)
                print(f"this should be 1,6 {add_time_ids.shape=}")
                return add_time_ids

            add_time_ids = jnp.concatenate(
                [
                    compute_time_ids(s, c)
                    for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ]
            )
            # Predict the noise residual
            unet_added_conditions = {"time_ids": add_time_ids}
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
            print(
                f"my: {add_time_ids.shape=}, {prompt_embeds.shape=}, {pooled_prompt_embeds.shape=}"
            )

            model_pred = unet.apply(
                {"params": params},
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                train=True,
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    noise_scheduler_state, model_input, noise, timesteps
                )
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            loss = (target - model_pred) ** 2
            loss = loss.mean()
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        # grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        # metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    print("replicating trainstep ...")
    # p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,), devices=jax_devices[1:2])

    # Replicate the train state on each device
    # print("replicating state ...")
    # state = jax_utils.replicate(state, devices=jax_devices[2:3])
    # print("replicating text_encoder_1_params ...")
    # text_encoder_1_params = jax_utils.replicate(text_encoder_1.params, devices=jax_devices[3:4])
    # print("replicating text_encoder_2_params ...")
    # text_encoder_2_params = jax_utils.replicate(text_encoder_2.params, devices=jax_devices[4:5])
    # print("replicating vae_params ...")
    # vae_params = jax_utils.replicate(vae_params, devices=jax_devices[5:6])

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0

    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_metrics = []

        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(
            total=steps_per_epoch, desc="Training...", position=1, leave=False
        )
        # train
        for batch in train_dataloader:
            # batch = shard(batch)
            # batch = jax.tree_util.tree_map(lambda x: x.reshape((total_train_batch_size, -1) + x.shape[1:]), batch)

            # state, train_metric, train_rngs = p_train_step(state, batch, train_rngs)
            state, train_metric, train_rngs = train_step(state, batch, train_rngs)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= args.max_train_steps:
                break

        # train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})"
        )

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        scheduler = FlaxPNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
        )
        pipeline = FlaxStableDiffusionXLPipeline(
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        pipeline.save_pretrained(
            args.output_dir,
            params={
                "text_encoder": get_params_to_save(text_encoder_1.params),
                "text_encoder_2": get_params_to_save(text_encoder_2.params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(state.params),
            },
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )


if __name__ == "__main__":
    main()
    # exc = None
    # with jax.profiler.trace("/mnt/disks/persist/repos/tensor_prf"):
    #     try:
    #         main()
    #     except Exception as e:
    #         exc = e
    #         print(e)

    # if exc is not None:
    #     raise exc
