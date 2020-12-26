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


def main(args):
    set_package_verbosity(args.debug)
    data_collection = data_factory.preprocess(args)

    with tf.Session(config=get_tf_config_proto()).as_default():
        signature = load_serving_signature(args.model_path / args.version_number)
        generator = PerplexityCalculator.from_signature(signature['perplexity'])
        for tag, dataset in data_collection.items():
            print(f"Evaluate {tag} perplexity:")
            perplexity = generator.perplexity(dataset.ids)
            print(f"Perplexity = {perplexity}")


def parse_args(argv):
    from flexparse import ArgumentParser
    from scripts.parsers import develop_parser, load_parser

    return ArgumentParser(parents=[
        data_factory.PARSER,
        develop_parser(),
        load_parser(),
    ]).parse_args(argv)


if __name__ == '__main__':
    import sys
    main(parse_args(sys.argv[1:]))
