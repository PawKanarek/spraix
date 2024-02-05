<div align="center">
  
![](https://raw.githubusercontent.com/PawKanarek/spraix/main/logo_spraix.jpeg) 

Spraix: making animated sprites with power of [Stable Diffusion](https://stability.ai/stable-diffusion).

Powered by Google TPU Research Cloud. 
</div>

# Goal 
To make animated sprites easily with the power of Stable Diffusion.

# Notes
- This repository if __fat__ because I pushed training dataset and evaluation ouptut, so keep in mind that cloning might take a while. I know it's bad, but this is my hobby project and sometimes I used the shortcuts. 
- This fine tuned model is far from perfect, It generates very, **very** ugly images. You can see them [here](https://github.com/PawKanarek/spraix/blob/48d8c209a359622e6db56e6d555667ac466dc952/output/sdxl-base-flax/sdxl_best_96_32.png) (Be patient, this is 90 MB image!)


# Done
- Gather data
- Label data
- Make training dataset
- Scripts that can transform raw data sprites into dataset that Stable Diffusion models can be trained on
- training SDXL with FLAX framework [train_text_to_image_flax_sdxl.py](https://github.com/PawKanarek/spraix/blob/main/train_text_to_image_flax_sdxl.py)
- Fine tune SDXL with my training
- Fix bugs in training data
- Make comparison of each epoch

# WIP 
- Share your failures with world

# TODO
- Different approach: Don't try to make sprite animations from a single image, make a video from image - that sounds like animation! https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt 
- LoRA training with FLAX
- Write the script to transform single image into sprite animation

# Setup
- install python dependencies with `pip install -r requirements.txt`
- You can run interference with [text_to_image.py](https://github.com/PawKanarek/spraix/blob/main/text_to_image.py)
- You can run fine tuning with [train_text_to_image_flax_sdxl.py](https://github.com/PawKanarek/spraix/blob/main/train_text_to_image_flax_sdxl.py) with params:
```bash 
--pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' --train_data_dir='train_data_1024_best_96/' --resolution=1024 --center_crop --train_batch_size=4 --mixed_precision='bf16' --num_train_epochs=16 --learning_rate=1e-05 --max_grad_norm=1 --output_dir='spraix_sdxl_best_96_16'
```
- This repository is fat because I attached interference images from each training step
- You can make a train dataset with [make_train_data.py](https://github.com/PawKanarek/spraix/blob/main/make_train_data.py) but you would have to manually download all the assets mentioned in `Training Data` section.

# Training Data
Special thanks to the skilled sprite animation creators, contributing to the training dataset for this project.

- Train images [0.png](train_data_1024/images/0.png) - [6.png](train_data_1024/images/6.png) thanks to https://oisougabo.itch.io/gap-i 
- Train images [7.png](train_data_1024/images/7.png) - [21.png](train_data_1024/images/21.png) thanks to https://szadiart.itch.io/2d-soulslike-character 
- Train images [22.png](train_data_1024/images/22.png) - [29.png](train_data_1024/images/29.png) thanks to https://admurin.itch.io/mega-admurins-freebies 
- Train images [30.png](train_data_1024/images/30.png) - [37.png](train_data_1024/images/37.png) thanks to https://astrobob.itch.io/arcane-archer 
- Train images [38.png](train_data_1024/images/38.png) - [43.png](train_data_1024/images/43.png) thanks to https://penusbmic.itch.io/sci-fi-character-pack-10 
- Train images [44.png](train_data_1024/images/44.png) - [44.png](train_data_1024/images/44.png) thanks to https://creativekind.itch.io/gif-bloodmoon-tower-free 
- Train images [45.png](train_data_1024/images/45.png) - [51.png](train_data_1024/images/51.png) thanks to https://clembod.itch.io/bringer-of-death-free 
- Train images [52.png](train_data_1024/images/52.png) - [71.png](train_data_1024/images/71.png) thanks to https://admurin.itch.io/mega-admurins-freebies 
- Train images [72.png](train_data_1024/images/72.png) - [97.png](train_data_1024/images/97.png) thanks to https://assetbakery.itch.io/2d-fighter-3 
- Train images [98.png](train_data_1024/images/98.png) - [102.png](train_data_1024/images/102.png) thanks to https://ansimuz.itch.io/dancing-girl-sprites 
- Train images [103.png](train_data_1024/images/103.png) - [126.png](train_data_1024/images/126.png) thanks to https://chierit.itch.io/elementals-leaf-ranger 
- Train images [127.png](train_data_1024/images/127.png) - [141.png](train_data_1024/images/141.png) thanks to https://chierit.itch.io/elementals-fire-knight 
- Train images [142.png](train_data_1024/images/142.png) - [157.png](train_data_1024/images/157.png) thanks to https://chierit.itch.io/elementals-water-priestess 
- Train images [158.png](train_data_1024/images/158.png) - [162.png](train_data_1024/images/162.png) thanks to https://luizmelo.itch.io/evil-wizard 
- Train images [163.png](train_data_1024/images/163.png) - [167.png](train_data_1024/images/167.png) thanks to https://penusbmic.itch.io/monster-pack-i 
- Train images [168.png](train_data_1024/images/168.png) - [169.png](train_data_1024/images/169.png) thanks to https://foozlecc.itch.io/void-environment-pack 
- Train images [170.png](train_data_1024/images/170.png) - [175.png](train_data_1024/images/175.png) thanks to https://xyezawr.itch.io/gif-free-pixel-effects-pack-6-forks-of-flame 
- Train images [176.png](train_data_1024/images/176.png) - [183.png](train_data_1024/images/183.png) thanks to https://luizmelo.itch.io/hero-knight-2 
- Train images [184.png](train_data_1024/images/184.png) - [191.png](train_data_1024/images/191.png) thanks to https://luizmelo.itch.io/hero-knight 
- Train images [192.png](train_data_1024/images/192.png) - [198.png](train_data_1024/images/198.png) thanks to https://luizmelo.itch.io/huntress-2 
- Train images [199.png](train_data_1024/images/199.png) - [208.png](train_data_1024/images/208.png) thanks to https://luizmelo.itch.io/huntress 
- Train images [209.png](train_data_1024/images/209.png) - [216.png](train_data_1024/images/216.png) thanks to https://luizmelo.itch.io/martial-hero-2 
- Train images [217.png](train_data_1024/images/217.png) - [225.png](train_data_1024/images/225.png) thanks to https://luizmelo.itch.io/martial-hero-3 
- Train images [226.png](train_data_1024/images/226.png) - [233.png](train_data_1024/images/233.png) thanks to https://luizmelo.itch.io/martial-hero 
- Train images [234.png](train_data_1024/images/234.png) - [242.png](train_data_1024/images/242.png) thanks to https://luizmelo.itch.io/medieval-king-pack-2 
- Train images [243.png](train_data_1024/images/243.png) - [252.png](train_data_1024/images/252.png) thanks to https://luizmelo.itch.io/medieval-warrior-pack-2 
- Train images [253.png](train_data_1024/images/253.png) - [261.png](train_data_1024/images/261.png) thanks to https://luizmelo.itch.io/medieval-warrior-pack-3 
- Train images [262.png](train_data_1024/images/262.png) - [278.png](train_data_1024/images/278.png) thanks to https://admurin.itch.io/pixel-character-horse-rider 
- Train images [279.png](train_data_1024/images/279.png) - [279.png](train_data_1024/images/279.png) thanks to https://mattwalkden.itch.io/free-robot-warfare-pack 
- Train images [280.png](train_data_1024/images/280.png) - [294.png](train_data_1024/images/294.png) thanks to https://szadiart.itch.io/rocky-world-platformer-set 
- Train images [295.png](train_data_1024/images/295.png) - [298.png](train_data_1024/images/298.png) thanks to https://penusbmic.itch.io/characterpack1 
- Train images [299.png](train_data_1024/images/299.png) - [302.png](train_data_1024/images/302.png) thanks to https://penusbmic.itch.io/monster-pack-i 
- Train images [303.png](train_data_1024/images/303.png) - [311.png](train_data_1024/images/311.png) thanks to https://darkpixel-kronovi.itch.io/undead-executioner 
- Train images [312.png](train_data_1024/images/312.png) - [319.png](train_data_1024/images/319.png) thanks to https://luizmelo.itch.io/wizard-pack 
- Train images [320.png](train_data_1024/images/320.png) - [324.png](train_data_1024/images/324.png) thanks to https://chierit.itch.io/boss-demon-slime 
- Train images [325.png](train_data_1024/images/325.png) - [384.png](train_data_1024/images/384.png) thanks to https://scrabling.itch.io/pixel-isometric-tiles 
- Train images [385.png](train_data_1024/images/385.png) - [389.png](train_data_1024/images/389.png) thanks to https://rili-xl.itch.io/cultist-priest-pack 
- Train images [390.png](train_data_1024/images/390.png) - [405.png](train_data_1024/images/405.png) thanks to https://arks.itch.io/dino-characters 
- Train images [406.png](train_data_1024/images/406.png) - [419.png](train_data_1024/images/419.png) thanks to https://chierit.itch.io/elementals-leaf-ranger 
- Train images [420.png](train_data_1024/images/420.png) - [423.png](train_data_1024/images/423.png) thanks to https://opengameart.org/content/lpc-maskman 
- Train images [424.png](train_data_1024/images/424.png) - [428.png](train_data_1024/images/428.png) thanks to https://penusbmic.itch.io/monster-pack-i 
- Train images [429.png](train_data_1024/images/429.png) - [431.png](train_data_1024/images/431.png) thanks to https://bdragon1727.itch.io/free-trap-platformer 
- Train images [432.png](train_data_1024/images/432.png) - [559.png](train_data_1024/images/559.png) thanks to https://github.com/YingzhenLi/Sprites 



