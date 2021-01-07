import os
import pathlib
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

from core.train.callbacks import ModelCheckpoint
from factories.trainer_factory.GAN import GANCreator
from factories.trainer_factory.MLE import MLECreator

from . import train


def main(args):
    restore_path = args.path
    main_args_path = restore_path / 'args'
    try:
        with open(main_args_path, 'r') as f_in:
            main_argv = f_in.read().split()
    except FileNotFoundError:
        raise FileNotFoundError(f"{main_args_path} not found, checkpoint can't be restored.")

    # HACK
    algorithm = GANCreator if 'GAN' in main_argv[0] else MLECreator
    train_args = train.parse_args(main_argv[1:], algorithm=algorithm)
    train_args.__dict__.update(args.__dict__)

    train.main(
        train_args,
        base_tag=os.path.basename(restore_path),
        checkpoint=ModelCheckpoint.latest_checkpoint(restore_path),
    )


def parse_args(argv):
    from flexparse import ArgumentParser, SUPPRESS, IntRange
    from scripts.parsers import save_parser

    parser = ArgumentParser(
        parents=[save_parser(argument_default=SUPPRESS)],
        argument_default=SUPPRESS,
    )
    parser.add_argument(
        'path',
        type=pathlib.Path,
        help="load checkpoint from this file prefix.",
    )
    parser.add_argument(
        '--epochs',
        type=IntRange(1),
        help="number of training epochs. (default: same as original args.)",
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(parse_args(sys.argv[1:]))
