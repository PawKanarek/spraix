import os
import re
import uuid

PROMPT = (
    "cook dinosaur walking down the forest to school with backpack to teach about proper eating"
)
NEGATIVE_PROMPT = "blurry, low res, low quality"
STEPS = 40
SEED = 420
GUIDANCE_SCALE = 5.0
REFINER_KICK_IN = 0.8
IMG_NAME = ""  # leave empty to auto generate
BASE_ID = "/mnt/disks/persist/repos/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "/mnt/disks/persist/repos/pawai/output/"
RUN_ID = str(uuid.uuid4())


def getSavePath(index: int = 0) -> str:
    name = IMG_NAME
    if not name:
        name = PROMPT

    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = name.replace(" ", "_")
    file_path = os.path.join(OUTPUT_DIR, name)
    dir, name = os.path.split(file_path)
    name = name[:200]
    name = f"{name}_{RUN_ID}_{index}.png"
    file_path = os.path.join(dir, name)

    print(f"saving to {file_path}")
    return file_path


def create_ouptut_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
