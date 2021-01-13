import os
import pathlib

from dotenv import load_dotenv

from flexparse import ArgumentParser, IntRange
from library.utils import format_path


load_dotenv('.env')


def evaluate_parser(**kwargs):
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('evaluate', description='Settings of evaluation metrics.')
    group.add_argument(
        '--bleu',
        nargs='?',
        type=IntRange(1, 5),
        const=5,
        help="longest n-gram to calculate BLEU/SelfBLEU score (5 if not specified).",
    )
    group.add_argument(
        '--fed',
        nargs='?',
        type=IntRange(minval=1),
        const=10000,
        help="number of sample size for FED score.",
    )
    return parser


def save_parser(**kwargs):
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('save', description="Settings of saving model.")
    group.add_argument(
        '--checkpoint-root', '--ckpt',
        type=pathlib.Path,
        help="save checkpoint to this directory.",
    )
    if group.get_default('checkpoint_root') is None:  # to avoid interfering SUPPRESS
        group.set_defaults(checkpoint_root=os.getenv('CHECKPOINT_DIR'))

    group.add_argument(
        '--serving-root',
        type=pathlib.Path,
        help='save serving model to this directory.',
    )
    group.add_argument(
        '--save-period',
        type=IntRange(minval=1),
        default=1,
        help="interval (number of epochs) between each saving.",
    )
    return parser


def load_parser(**kwargs):
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('load', description='Settings of loading saved model.')
    group.add_argument(
        '--model-path',
        type=pathlib.Path,
        required=True,
        help='path of serving model folder.',
    )
    return parser


def develop_parser():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group('develop', description='Developers only options.')
    group.add_argument(
        '--profile',
        nargs='?',
        type=pathlib.Path,
        const='./profile_stats',
        help=f"export profile stats to file ({format_path('./profile_stats')} if not specified).",
    )
    return parser


def train_parser():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group('train', description="Training parameters.")
    group.add_argument(
        '--epochs',
        type=IntRange(minval=1),
        default=10000,
        help="number of training epochs.",
    )
    group.add_argument(
        '--batch-size',
        type=IntRange(minval=1),
        default=64,
        help="size of data mini-batch.",
    )
    group.add_argument(
        '--random-seed', '--seed',
        type=int,
        help='the global random seed.',
    )
    return parser


def logging_parser():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group('logging', description="Settings of logging.")
    group.add_argument(
        '--tensorboard',
        nargs='?',
        type=pathlib.Path,
        const=os.getenv('TENSORBOARD_LOGDIR'),
        help="whether to log experiment on tensorboard "
             "(os.env['TENSORBOARD_LOGDIR'] if not specified).",
    )
    group.add_argument(
        '--tags',
        nargs='+',
        metavar='TAG',
        default=[],
        help="additional tags to configure this training (will be used in tensorboard).",
    )
    return parser
