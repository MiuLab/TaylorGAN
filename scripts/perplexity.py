import os
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf

from core.evaluate import PerplexityCalculator
from factories import data_factory
from scripts.snippets import (
    get_tf_config_proto,
    set_package_verbosity,
    load_serving_signature,
)


def main(argv):
    args = parse_args(argv)
    set_package_verbosity(args.debug)
    data_collection = data_factory.preprocess(args)

    with tf.Session(config=get_tf_config_proto()).as_default():
        signature = load_serving_signature(os.path.join(args.model_path, args.version_number))
        generator = PerplexityCalculator.from_signature(signature['perplexity'])
        for tag, dataset in data_collection.items():
            print(f"Evaluate {tag} perplexity:")
            perplexity = generator.perplexity(dataset.ids)
            print(f"Perplexity = {perplexity}")


def parse_args(argv):
    from library.my_argparse import MyArgumentParser
    from scripts.core_parsers import develop_parser, load_parser

    return MyArgumentParser(parents=[
        data_factory.create_parser(),
        develop_parser(),
        load_parser(),
    ]).parse_args(argv)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
