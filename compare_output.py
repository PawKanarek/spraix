import os
import re
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_DIR = "output/sdxl-base-flax/spraix_sdxl_best_96_16"


def sort_by_number(a):
    # extract the last numbers & and sort
    pattern = re.compile(r"(\d+)(?=\D*$)")
    return sorted(a, key=lambda x: int(pattern.search(x).group(1)))


if __name__ == "__main__":
    imgs = [f for f in os.listdir(OUTPUT_DIR) if ".png" in f]
    dirs = [d for d in os.listdir(OUTPUT_DIR) if "_epoch_" in d]
    dirs = dirs[:2]
    # wspaces = [0.3 if i % 8 == 0 else 0 for i, _ in enumerate(imgs)]
    # hspaces = [0.05 for i, _ in enumerate(imgs)]
    fig, axs = plt.subplots(len(dirs), len(imgs))
    # plot all epochs
    for i, dir in enumerate(sort_by_number(dirs)):
        imgs = os.listdir(os.path.join(OUTPUT_DIR, dir))

        for j, img_path in enumerate(sort_by_number(imgs)):
            row = len(dirs) - i - 1
            ax = axs[row, j]
            img = Image.open(os.path.join(OUTPUT_DIR, dir, img_path))
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if j % 8 == 0:
                # remove hash, img number and .png from img path
                ax.set_title(img_path[:-43].replace("_", " "), loc="left")

            if j == 0:
                print(f"set label:{dir} to row:{row}, column:{j}")
                ax.set_ylabel(dir)
                ax.get_yaxis().set_visible(True)
                ax.set_yticklabels([])

    fig.subplots_adjust(left=0.01, top=1, right=1, bottom=0, wspace=0.05, hspace=0)
    fig.savefig("cmp.png")
    print("done")
