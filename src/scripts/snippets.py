import os
import random
import re

import numpy as np
import torch


def get_subdir_if_unique_base_tag_exists(path):
    dirs = os.listdir(path)
    if len(dirs) == 1 and re.match(r'.+@\d{8}-\d{6}', dirs[0]):
        return os.path.join(path, dirs[0])
    return path


def set_package_verbosity(debug):
    if not debug:
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.logging.set_verbosity(tf.logging.ERROR)


def set_global_random_seed(seed: int):
    print(f"seed = {seed}")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
