import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

OUTPUT_DIR = "output/sdxl-base-flax/spraix_sdxl_best_96_16"


def sort_by_number(a):
    # extract the last numbers & and sort
    pattern = re.compile(r"(\d+)(?=\D*$)")
    return reversed(sorted(a, key=lambda x: int(pattern.search(x).group(1))))


if __name__ == "__main__":
    imgs = [f for f in os.listdir(OUTPUT_DIR) if ".png" in f]
    dirs = [d for d in os.listdir(OUTPUT_DIR) if "_epoch_" in d]
    scale = 64
    fig = plt.figure(figsize=(2 * scale, 1 * scale))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(dirs), len(imgs)), axes_pad=0.1)

    for row, dir in enumerate(sort_by_number(dirs)):
        imgs = os.listdir(os.path.join(OUTPUT_DIR, dir))
        for j, img_path in enumerate(sort_by_number(imgs)):
            index = (row * len(imgs)) + j
            ax = grid[index]
            print(f"{img_path=}, {index=}, {row=}")
            img = Image.open(os.path.join(OUTPUT_DIR, dir, img_path))
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if row == 0 and j % 8 == 0:
                # remove hash, img number and .png from img path
                ax.set_title(img_path[:-43].replace("_", " "), loc="left")

            if j == 0:
                print(f"set label:{dir} to row:{row}, column:{j}")
                ax.set_ylabel(dir)
                ax.get_yaxis().set_visible(True)
                ax.set_yticklabels([])

    fig.subplots_adjust(
        left=0.005, top=0.995, right=0.995, bottom=0.005, wspace=0.005, hspace=0.005
    )
    fig.savefig("cmp.png")
    print("done")
