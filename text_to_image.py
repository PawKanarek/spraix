import time

from generate import with_sd_14_finetune

# from generate import with_pytorch

if __name__ == "__main__":
    start_time = time.time()
    with_sd_14_finetune.run()
    # with_pytorch.run()
    print(f"total time: {time.time() - start_time}")
