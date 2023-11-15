import time

from generate import common, with_flax, with_pytorch

if __name__ == "__main__":
    start_time = time.time()
    common.create_ouptut_dirs()
    with_flax.run()
    with_pytorch.run()
    print(f"total time: {time.time() - start_time}")
