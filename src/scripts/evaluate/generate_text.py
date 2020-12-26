import pathlib
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf

from core.evaluate import TextGenerator
from core.preprocess import Tokenizer
from scripts.snippets import set_package_verbosity, get_tf_config_proto, load_serving_signature


def main(args):
    set_package_verbosity(args.debug)

    with tf.Session(config=get_tf_config_proto()).as_default():
        signature = load_serving_signature(args.model_path / args.version_number)
        tokenizer = Tokenizer.load(args.model_path.parent / 'tokenizer.json')
        generator = TextGenerator.from_signature(signature['generate'], tokenizer=tokenizer)
        print(f"Generate sentences to '{args.export_path}'")
        with open(args.export_path, 'w') as f_out:
            f_out.writelines([
                sentence + "\n"
                for sentence in generator.generate_texts(args.samples)
            ])


def parse_args(argv):
    from flexparse import ArgumentParser, IntRange
    from scripts.parsers import develop_parser, load_parser

    parser = ArgumentParser(parents=[develop_parser(), load_parser()])
    parser.add_argument(
        '--export-path',
        type=pathlib.Path,
        required=True,
        help='path to save generated texts.',
    )
    parser.add_argument(
        '--samples',
        type=IntRange(minval=1),
        default=10000,
        help='number of generated samples(sentences).',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(parse_args(sys.argv[1:]))
