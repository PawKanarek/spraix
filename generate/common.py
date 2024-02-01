import os
import re
import uuid

PROMPT = "12-frame sprite animation of: cute small dinosaur with backpack, that: is running, facing: East"
NEGATIVE_PROMPT = "blurry, low res, low quality"
STEPS = 40
SEED = 420
GUIDANCE_SCALE = 5.0
REFINER_KICK_IN = 0.8
IMG_NAME = ""  # leave empty to auto generate
BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "output/sdxl-base-flax"
RUN_ID = str(uuid.uuid4())


def getSavePath(prompt: str = PROMPT, index: int = 0, sub_dir: str = "") -> str:
    dir = create_ouptut_dirs(sub_dir)

    name = IMG_NAME or prompt or PROMPT

    name = re.sub(r"[^a-zA-Z0-9\s]", "", str(name))
    name = name.replace(" ", "_")
    file_path = os.path.join(dir, name)
    dir, name = os.path.split(file_path)
    name = name[:200]
    name = f"{name}_{RUN_ID}_{index}.png"
    file_path = os.path.join(dir, name)

    print(f"saving to {file_path}")
    return file_path


def create_ouptut_dirs(sub_dir: str = "") -> str:
    dir = os.path.join(OUTPUT_DIR, sub_dir) if sub_dir else OUTPUT_DIR
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
