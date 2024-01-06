import os
import json

FOLDERS = ["train_data_512", "train_data_1024", "train_data_1024_16", "train_data_1024_best", "train_data_1024_best_96"]

if __name__ == "__main__":
    
    for folder in FOLDERS:
        all_images = os.listdir(os.path.join(folder, "images"))
        with open(os.path.join(folder, "metadata.jsonl"), "r+") as f:
            lines_to_save = []
            for line in f:
                if any(os.path.join("images", image) in line for image in all_images):
                    lines_to_save.append(line)
                else:
                    print(f'removing line: {line}')
            
            f.seek(0)
            f.truncate()
            f.writelines(lines_to_save)