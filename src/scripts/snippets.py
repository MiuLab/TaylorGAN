import os
import random
import re

import numpy as np
import tensorflow as tf


def get_subdir_if_unique_base_tag_exists(path):
    dirs = os.listdir(path)
    if len(dirs) == 1 and re.match(r'.+@\d{8}-\d{6}', dirs[0]):
        return os.path.join(path, dirs[0])
    return path


def set_package_verbosity(debug):
    if not debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.logging.set_verbosity(tf.logging.ERROR)


def set_global_random_seed(seed):
    print(f"seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def get_tf_config_proto(jit: bool = False):
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    if jit:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    return config


def load_serving_signature(path):
    print(f"Load model from '{path}'")
    meta_graph_def = tf.saved_model.loader.load(
        sess=tf.get_default_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir=str(path),
    )
    return meta_graph_def.signature_def
