import glob
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from os import path
from typing import Dict, List

import numpy as np
from dataclass_wizard import JSONWizard
from PIL import Image

# to work with this script u need to download manually sprites mentioned in 'credits.md', Authors don't allow raw redistribution of sprites, so i cannot put them into this reposisotry.


@dataclass
class SequenceData(JSONWizard):
    desc: str
    path: str | None = None
    w: int | None = None
    h: int | None = None
    frames: int | None = None
    first: int | None = None
    last: int | None = None
    row: int | None = None
    startIndex: int | None = None
    character: str | None = None
    row: int | None = None
    side: str | None = None


@dataclass
class SpriteData(JSONWizard):
    style: str
    background: str
    type: str
    seq: list[SequenceData]
    url: str | None = None
    license: str | None = None
    w: int | None = None
    h: int | None = None
    character: str | None = None
    side: str | None = None


@dataclass
class ResizeData:
    rows: int
    columns: int
    w: int
    h: int


# shoes and eyes don't differ too much so ignore
class Attr(Enum):
    body = 1
    bottomwear = 2
    # eyes = 3
    hair = 4
    # shoes = 5
    topwear = 6
    action = 7


class Actions(Enum):
    spell = 1
    dance = 2
    walk = 3
    slash = 4
    shot = 5
    fall = 6


# globals
metadata_content = []
credits_content = []
img_index: int = 0

SIZE = 1024
FRAMES_LIMIT = 9  # set -1 to add everything
MAKE_SQUARE = True
F_DESC = f"_f_{FRAMES_LIMIT}" if FRAMES_LIMIT > 0 else ""
DATA_DIR = f"train_data_{SIZE}{F_DESC}"
RAW_DIR = "raw_sprites"
COMPOSITE_DIR = "raw_composite"
SPRITE_DATA_FILE = "_sprite_data.json"
IMAGES_DIR = "images"
ID_ESCAPE_CHAR = "{id}"
METADATA_PATH = path.join(DATA_DIR, "metadata.jsonl")
DATA_IMAGES_DIR = path.join(DATA_DIR, IMAGES_DIR)
COMPOSITE_RAND_CHARS = 0


ACTIONS_DESC = {
    Actions.spell: "casts a spell",
    Actions.dance: "dances",
    Actions.walk: "walks",
    Actions.slash: "is slashing",
    Actions.shot: "shoots from invisible bow",
    Actions.fall: "falls",
}

SIDES_DESC = {
    "N": "North",
    "NE": "North-East",
    "E": "East",
    "SE": "South-East",
    "S": "South",
    "SW": "South-West",
    "W": "West",
    "NW": "North-West",
}

COMPOSITE_ATTR = {
    Attr.body: {
        "0": "a man",
        "1": "a man with skin of yellowish-orange color",
        "2": "a man with skin of teal-blue color",
        "3": "a man with pale white skin",
        "4": "a man with skin of dark-orange color",
        "5": "a man with brown skin",
        "6": "an orc",
    },
    Attr.hair: {
        "0": " with short green hair",
        "1": " with purple man bun hair",
        "2": " with short yellow hair",
        "3": " with short gray hair",
        "4": " with short pinkish-red hair",
        "5": " with curly pinkish-purple hair",
        "6": " with half-shaven gray hair",
        "7": " with short red hair",
        "8": " with short pink hair",
        "9": " with curly orange hair",
        "no": " without hair",
    },
    Attr.bottomwear: {
        "0": " in white shorts",
        "1": " in leather pants",
        "2": " in red shorts",
        "3": " in white pants",
        "4": " in green shorts",
        "5": " in dark green pants",
        "6": " in leather sash",
        # "no": " without pants",
    },
    Attr.topwear: {
        "0": " and red shirt",
        "1": " and blue shirt",
        "2": " and white shirt",
        "3": " and gray armor",
        "4": " and leather armor",
        # "5": " and white formal shirt with tie", # <- Incomplete data for shooting
        "6": " and gray chainmail",
        "no": " and without shirt",
    },
    # Attribute.shoes: {
    #     "0": " and brown shoes",
    #     "1": " wearing yellow shoes",
    #     "2": " wearing white shoes",
    #     "no": " barefoot",
    # },
    # ignore eyes, they don't change character so much so there is little training value
    # Attribute.eyes: {
    #     "0": "blue eyes",
    #     "1": "brown eyes",
    #     "2": "red eyes",
    #     "3": "green eyes",
    #     "4": "yellow eyes",
    #     "no": "black eyes",
    # },
}

COMPOSITE_ACTIONS = {
    Actions.spell: {
        "N": list(range(0, 7)),
        "W": list(range(13, 20)),
        "S": list(range(26, 33)),
        "E": list(range(39, 46)),
    },
    Actions.dance: {
        "N": list(range(52, 60)),
        "W": list(range(65, 73)),
        "S": list(range(78, 86)),
        "E": list(range(91, 99)),
    },
    Actions.walk: {
        "N": list(range(104, 113)),
        "W": list(range(117, 126)),
        "S": list(range(130, 139)),
        "E": list(range(143, 152)),
    },
    Actions.slash: {
        "N": list(range(156, 162)),
        "W": list(range(169, 175)),
        "S": list(range(182, 188)),
        "E": list(range(195, 201)),
    },
    Actions.shot: {
        "N": list(range(208, 221)),
        "W": list(range(221, 234)),
        "S": list(range(234, 247)),
        "E": list(range(247, 260)),
    },
    Actions.fall: {"S": list(range(260, 266))},
}


# define preset attributes that will result most diverse set
COMPOSITE_PRESETS = {
    Attr.body: ["0", "1", "3", "5", "6"],
    Attr.bottomwear: ["1", "2", "3", "4", "6"],
    Attr.hair: ["0", "1", "3", "5", "6", "7", "9", "no"],
    Attr.topwear: ["1", "3", "4", "5", "6", "no"],
}


def save_img(sprite, seq, sequence_img):
    global img_index
    img_name = f"{img_index}.png"
    img_index += 1
    append_metadata(img_name, seq, sprite)
    sequence_img.save(path.join(DATA_IMAGES_DIR, img_name))


def get_composite_imgs() -> Dict[Attr, Dict[str, Image.Image]]:
    # create dictionary of all attribute variants; structure: attr_img = {'body': {'0': Img, ...}, ...}
    attr_img: Dict[Attr, Dict[str, Image.Image]] = {}
    for attr_name, attr_values in COMPOSITE_ATTR.items():
        attr_img[attr_name] = {}
        for name in attr_values:
            if name == "no":
                attr_img[attr_name][name] = Image.NONE
                continue
            img_path = os.path.join(COMPOSITE_DIR, attr_name.name, f"{name}.png")
            attr_img[attr_name][name] = Image.open(img_path)

    return attr_img


def get_preset_random_characters() -> List[Dict[Attr, Dict[str, object]]]:
    characters = []
    character = {}
    for attr, values in COMPOSITE_PRESETS.items():
        rnd_value = np.random.choice(values)
        character[attr] = {rnd_value: COMPOSITE_ATTR[attr][rnd_value]}

    # remove hair and shirt for an orc
    if character[Attr.body] == "6":
        character.pop(Attr.hair)
        character.pop(Attr.topwear)

    rnd_action_key = np.random.choice([Actions.shot, Actions.walk])
    rnd_action_val = COMPOSITE_ACTIONS[rnd_action_key]

    for side_key, side_val in rnd_action_val.items():
        character[Attr.action] = {rnd_action_key: {side_key: side_val}}
        characters.append(character.copy())
    return characters


def get_preset_charters() -> List[Dict[Attr, Dict[str, object]]]:
    # create 16 unique & diverse characters - done manually for reproducibility
    # this will result in 16 * 4(sides) * 2(actions) = 128 unique sprites
    preset = [
        ["0", "1", "0", "4"],
        ["0", "2", "1", "6"],
        ["0", "3", "3", "no"],
        ["1", "2", "5", "0"],
        ["1", "3", "6", "1"],
        ["1", "4", "7", "2"],
        ["2", "3", "9", "3"],
        ["3", "0", "no", "4"],
        ["3", "6", "0", "6"],
        ["3", "4", "1", "no"],
        ["4", "6", "3", "0"],
        ["5", "1", "5", "1"],
        ["5", "6", "6", "2"],
        ["5", "1", "7", "3"],
        ["6", "3", "no", "no"],
        ["6", "2", "no", "no"],
    ]
    characters = []
    for body, bottomwear, hair, topwear in preset:
        char = {}
        char[Attr.body] = {body: COMPOSITE_ATTR[Attr.body][body]}
        char[Attr.bottomwear] = {
            bottomwear: COMPOSITE_ATTR[Attr.bottomwear][bottomwear]
        }
        char[Attr.hair] = {hair: COMPOSITE_ATTR[Attr.hair][hair]}
        char[Attr.topwear] = {topwear: COMPOSITE_ATTR[Attr.hair][topwear]}

        for action in [Actions.shot, Actions.walk]:
            for side_key, side_val in COMPOSITE_ACTIONS[action].items():
                char[Attr.action] = {action: {side_key: side_val}}
                characters.append(char.copy())

    return characters


def cleanup_dirs():
    # remove all files
    if path.exists(DATA_IMAGES_DIR):
        imgs_path = path.join(DATA_IMAGES_DIR, "*")
        print(f"removing all {imgs_path}")
        imgs = glob.glob(imgs_path)
        for i in imgs:
            os.remove(i)
        print(f"removed {len(imgs)} old images")
    else:
        os.makedirs(DATA_IMAGES_DIR)


def get_resize(w: int, h: int, frames_n: int) -> ResizeData:
    print(frames_n)

    if FRAMES_LIMIT > 0:
        if frames_n < FRAMES_LIMIT:
            return
        frames_n = FRAMES_LIMIT if frames_n > FRAMES_LIMIT else frames_n

    if MAKE_SQUARE:
        if w > h:
            h = w
        if h > w:
            w = h

    total_area = frames_n * w * h
    max_area = SIZE * SIZE
    scaling_factor = (max_area / total_area) ** 0.5
    scaled_W = int(round(w * scaling_factor))
    scaled_H = int(round(h * scaling_factor))

    num_columns_by_width = math.ceil(SIZE / scaled_W)
    num_rows = math.ceil(frames_n / num_columns_by_width)
    num_columns = math.ceil(frames_n / num_rows)

    # Recalculate the actual width and height based on the adjusted number of columns
    scaled_W = int(SIZE / num_columns)
    scaled_H = int(scaled_W * (h / w))

    # Check if scaled_H exceeds the height constraint
    if scaled_H * num_rows > SIZE:
        scaled_H = int(SIZE / num_rows)
        scaled_W = int(scaled_H * (w / h))

    assert num_columns * scaled_W <= SIZE
    assert num_rows * scaled_H <= SIZE

    return ResizeData(num_rows, num_columns, scaled_W, scaled_H)


def add_frame_to_sequence(
    sequence_img: Image.Image,
    resize: ResizeData,
    row_i: int,
    column_i: int,
    frame_img: Image.Image,
):
    x = column_i * resize.w
    # y_offset = 0 if resize.rows == 1 else (row_i * resize.h)
    y = row_i * resize.h
    resized = frame_img.resize((resize.w, resize.h), resample=Image.BOX)
    sequence_img.alpha_composite(resized, (x, y))


def process_individual_frames(
    seq: SequenceData, sprite: SpriteData, root_dir: str
) -> Image.Image:
    sequence_img = Image.new("RGBA", (SIZE, SIZE), "gray")

    frames = list(range(seq.first, seq.last + 1))
    w = seq.w or sprite.w
    h = seq.h or sprite.h
    resize = get_resize(w, h, len(frames))
    if not resize:
        return
    frame_i = frames[0]
    # print(f"individual {name=}, {sequence_def=}")
    for row_i in range(resize.rows):
        for column_i in range(resize.columns):
            if frame_i > frames[-1]:
                continue
            frame_name = os.path.join(
                root_dir, seq.path.replace(ID_ESCAPE_CHAR, str(frame_i))
            )
            frame_i += 1
            frame = Image.open(frame_name).convert("RGBA")
            add_frame_to_sequence(sequence_img, resize, row_i, column_i, frame)
    return sequence_img


def crop_into_grid(img: Image.Image, w: int, h: int) -> Image.Image:
    img_w, img_h = img.size
    # print(f"{img_w=}, {img_h=}, {w=}, {h=}, {img_w // w}, {img_h // h}")
    rows = img_h // h
    columns = img_w // w
    assert rows > 0, f"probably {h=} in {SPRITE_DATA_FILE} is wrong"
    assert columns > 0, f"probably {w=} in {SPRITE_DATA_FILE} is wrong"
    for i in range(rows):
        for j in range(columns):
            box = (j * w, i * h, (j + 1) * w, (i + 1) * h)
            yield img.crop(box)


def process_sequence_frames(
    seq: SequenceData, sprite: SpriteData, root_dir: str, img: Image.Image = None
) -> Image.Image:
    sequence_img = Image.new("RGBA", (SIZE, SIZE), "gray")
    frames_def = list(range(seq.frames))
    w = seq.w or sprite.w
    h = seq.h or sprite.h
    resize = get_resize(w, h, len(frames_def))
    if not resize:
        return

    if not img:
        img_path = os.path.join(root_dir, seq.path)
        img = Image.open(img_path).convert("RGBA")
    frames_img = [i for i in crop_into_grid(img, w, h)]

    frame_i = 0
    ending_i = frames_def[-1]

    # if in seq item there is startIdex field, that means all frames are in single img
    if seq.startIndex is not None:
        frame_i = seq.startIndex
        ending_i = frame_i + len(frames_def) - 1

    # if in seq item there is row field, that means all frames are in single img and there is no other sequence
    if seq.row is not None:
        frame_i = int(seq.row * len(frames_img) / len(sprite.seq))
        ending_i = frame_i + len(frames_def) - 1

    # print(f"sequence: {frame_i=}, {ending_i=} {name_def=}, {sequence_def=}, {len(frames_img)=}")
    for row_i in range(resize.rows):
        for column_i in range(resize.columns):
            if frame_i > ending_i:
                continue
            frame = frames_img[frame_i]
            frame_i += 1
            add_frame_to_sequence(sequence_img, resize, row_i, column_i, frame)
    return sequence_img


def append_metadata(img_name: str, seq: SequenceData, sprite: SpriteData):
    char = seq.character or sprite.character
    frames = 0
    if seq.frames is not None:
        frames = seq.frames
    elif seq.first is not None:
        assert (
            seq.last is not None
        ), f"In {SPRITE_DATA_FILE} for {img_name} in {seq.desc} value of 'last' is missing!"
        frames = len(range(seq.first, seq.last + 1))

    side = seq.side or sprite.side or ""
    if side:
        side = f", facing: {SIDES_DESC[side]}"

    # for now ignore size: size_desc = of size ({resize.w}x{resize.h})
    # desc = f"Sprite animation of {char} that {seq.desc}. Made of {frames} frames{side}, {sprite.style}, gray background"
    if FRAMES_LIMIT > 0:
        desc = f"Pixel-art animation of {char}, that: {seq.desc}{side}"
    else:
        desc = f"{frames}-frame sprite animation of: {char}, that: {seq.desc}{side}"
    metadata_content.append(
        {"file_name": os.path.join(IMAGES_DIR, img_name), "text": desc}
    )


def name_of(dict) -> str:
    return next(iter(dict.items()))[0]


def description_of(dict) -> str:
    return next(iter(dict.items()))[1]


def img_link(i):
    name = f"{i}.png"
    return f"[{name}]({path.join(DATA_IMAGES_DIR, name)})"


def generate_composite() -> List[Dict[Attr, Dict[str, object]]]:
    global img_index
    begin_i = img_index
    unique_characters = []
    cached_imgs = get_composite_imgs()

    if COMPOSITE_RAND_CHARS > 0:
        while len(unique_characters) < COMPOSITE_RAND_CHARS:
            characters = get_preset_random_characters()
            for char in characters:
                if char not in unique_characters:
                    unique_characters.append(char)
    else:
        unique_characters = get_preset_charters()

    for i_char, character in enumerate(unique_characters):  # or random_characters
        body_img = cached_imgs[Attr.body][name_of(character[Attr.body])]
        bottomwear_img = cached_imgs[Attr.bottomwear][
            name_of(character[Attr.bottomwear])
        ]
        hair_img = cached_imgs[Attr.hair][name_of(character[Attr.hair])]
        topwear_img = cached_imgs[Attr.topwear][name_of(character[Attr.topwear])]
        imgs = [body_img, bottomwear_img, hair_img, topwear_img]

        body_desc = description_of(character[Attr.body])
        bottomwear_desc = description_of(character[Attr.bottomwear])
        hair_desc = description_of(character[Attr.hair])
        topwear_desc = description_of(character[Attr.topwear])
        action_desc, frames_dict = next(iter(character[Attr.action].items()))
        action_desc = ACTIONS_DESC[action_desc]
        side_desc, row_frames = next(iter(frames_dict.items()))

        isOrc = name_of(character[Attr.body]) == "6"
        if isOrc:
            imgs.remove(hair_img)
            imgs.remove(topwear_img)
            hair_desc = ""
            topwear_desc = ""

        # bg_color = np.random.choice(list(ImageColor.colormap.keys()))
        bg_color = "transparent"
        frame_size = 64
        max_frames_per_row = 13
        sequence_img = Image.new(
            "RGBA", (frame_size * len(row_frames), frame_size), (255, 0, 0, 0)
        )

        frame_row = int(row_frames[0] / max_frames_per_row)
        for i, frame_i in enumerate(row_frames):
            box = (
                i * frame_size,
                frame_row * frame_size,
                (i + 1) * frame_size,
                (frame_row + 1) * frame_size,
            )
            frame = Image.new("RGBA", (frame_size, frame_size), (255, 0, 0, 0))
            # stack each attribute into main frame
            for img in imgs:
                if img is Image.NONE:
                    continue
                attr_frame = img.crop(box).convert("RGBA")
                frame.alpha_composite(attr_frame)
            sequence_img.alpha_composite(frame, (i * frame_size, 0))

        seq = SequenceData(action_desc, frames=len(row_frames))
        sprite = SpriteData(
            w=64,
            h=64,
            style="pixel art style",
            background="transparent",
            type="sprite animation",
            character=f"{body_desc}{hair_desc}{bottomwear_desc}{topwear_desc}",
            side=side_desc,
            seq=[seq],
        )

        img = process_sequence_frames(seq, sprite, COMPOSITE_DIR, img=sequence_img)
        save_img(sprite, seq, img)
    last_index = img_index - 1
    credits_content.append(
        f"Train images {img_link(begin_i)} - {img_link(last_index)} thanks to https://github.com/YingzhenLi/Sprites"
    )
    print(f"saved {begin_i} - {last_index} composite characters")


def save_metadata():
    with open(METADATA_PATH, "w") as json_file:
        for item in metadata_content:
            json.dump(item, json_file)
            json_file.write("\n")
        print(f"Saved {json_file.name}")


def save_credits():
    with open("credits.md", "w") as md_file:
        md_file.write(
            """# Training Data
Special thanks to the skilled sprite animation creators for providing their work, contributing to the training dataset for this project.\n
"""
        )
        for item in credits_content:
            md_file.write(f"- {item} \n")

        print(f"Saved {md_file.name}")


def generate_dataset():
    # check if all dirs are and described correctly
    for dir in os.listdir(RAW_DIR):
        sprite_data = os.path.join(RAW_DIR, dir, SPRITE_DATA_FILE)
        if not os.path.exists(sprite_data) and dir != ".DS_Store":
            print(f"cannot find {SPRITE_DATA_FILE} in {dir}")

    for sprite_file in sorted(glob.glob(path.join(RAW_DIR, "*", SPRITE_DATA_FILE))):
        # for sprite_file in glob.glob(path.join(raw_dir, "critters", sprite_data_json)):
        with open(sprite_file, "r") as file:
            json_dict = json.load(file)
            sprite = SpriteData.from_dict(json_dict)

            assert sprite.url, "missing source url"
            assert sprite.license, "missing license info"
            root_dir = path.dirname(file.name)
            global img_index
            last_index = img_index + len(sprite.seq) - 1

            credits_content.append(
                f"Train images {img_link(img_index)} - {img_link(last_index)} thanks to {sprite.url}"
            )

            print(f"from {file.name} saving {img_index} - {last_index} sprites")
            for seq in sprite.seq:
                if ID_ESCAPE_CHAR in seq.path:
                    img = process_individual_frames(seq, sprite, root_dir)
                else:
                    img = process_sequence_frames(seq, sprite, root_dir)

                if img:
                    save_img(sprite, seq, img)


if __name__ == "__main__":
    cleanup_dirs()
    generate_dataset()
    generate_composite()
    save_metadata()
    save_credits()
