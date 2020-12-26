import os
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf

from scripts import train
from scripts.snippets import get_subdir_if_unique_base_tag_exists


def main(argv):
    restore_args = parse_args(argv)
    restore_path = get_subdir_if_unique_base_tag_exists(restore_args.path)
    main_args_path = os.path.join(restore_path, 'args')
    if not os.path.isfile(main_args_path):
        raise FileNotFoundError(f"{main_args_path} not found, checkpoint can't be restored.")
    with open(main_args_path, 'r') as f_in:
        main_argv = f_in.read().split()

    train.main(
        main_argv,
        base_tag=os.path.basename(restore_path),
        checkpoint=tf.train.latest_checkpoint(restore_path),
        override_namespace=restore_args,
    )


def parse_args(argv):
    from library.my_argparse import MyArgumentParser, SUPPRESS
    from library.my_argparse.types import int_in_range, path
    from scripts.core_parsers import save_parser

    parser = MyArgumentParser(
        parents=[save_parser(argument_default=SUPPRESS)],
        argument_default=SUPPRESS,
    )
    parser.add_argument(
        'path',
        type=path,
        help="Load checkpoint from this file prefix.",
    )
    parser.add_argument(
        '--epochs',
        type=int_in_range(1),
        help="Number of training epochs. (default: same as original args.)",
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(argv=sys.argv[1:])
