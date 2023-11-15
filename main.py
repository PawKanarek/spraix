import time

from generate import common, with_flax

if __name__ == "__main__":
    start_time = time.time()
    common.create_ouptut_dirs()
    with_flax.run()
    print(f"total time: {time.time() - start_time}")
