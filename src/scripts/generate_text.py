import os
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf

from core.evaluate import TextGenerator
from core.preprocess import Tokenizer
from scripts.snippets import set_package_verbosity, get_tf_config_proto, load_serving_signature


def main(argv):
    args = parse_args(argv)
    set_package_verbosity(args.debug)

    with tf.Session(config=get_tf_config_proto()).as_default():
        signature = load_serving_signature(os.path.join(args.model_path, args.version_number))
        tokenizer = Tokenizer.load(os.path.join(os.path.dirname(args.model_path), 'tokenizer.json'))
        generator = TextGenerator.from_signature(signature['generate'], tokenizer=tokenizer)
        print(f"Generate sentences to '{args.export_path}'")
        with open(args.export_path, 'w') as f_out:
            f_out.writelines([
                sentence + "\n"
                for sentence in generator.generate_texts(args.samples)
            ])


def parse_args(argv):
    from library.my_argparse import MyArgumentParser
    from library.my_argparse.types import path, int_in_range
    from scripts.core_parsers import develop_parser, load_parser

    parser = MyArgumentParser(parents=[develop_parser(), load_parser()])
    parser.add_argument(
        '--export-path',
        type=path,
        required=True,
        help='path to save generated texts.',
    )
    parser.add_argument(
        '--samples',
        type=int_in_range(minval=1),
        default=10000,
        help='number of generated samples(sentences).',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(argv=sys.argv[1:])
