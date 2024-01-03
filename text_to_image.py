import time

from generate import with_sdxl_finetune

# from generate import with_flax

if __name__ == "__main__":
    start_time = time.time()
    # with_flax.run()
    with_sdxl_finetune.run()
    # with_pytorch.run()
    print(f"total time: {time.time() - start_time}")
