import time

from generate import common, with_sd_14_finetune

if __name__ == "__main__":
    start_time = time.time()
    with_sd_14_finetune.run()
    print(f"total time: {time.time() - start_time}")
