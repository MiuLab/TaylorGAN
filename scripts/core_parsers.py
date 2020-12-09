import os

from dotenv import load_dotenv

from library.my_argparse import MyArgumentParser
from library.my_argparse.types import path, int_in_range


load_dotenv('.env')


def evaluate_parser(**kwargs):
    parser = MyArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('evaluate', description='Settings of evaluation metrics.')
    group.add_argument(
        '--bleu',
        nargs='?',
        type=int_in_range(1, 5),
        const=5,
        help="Max number of grams to calculate BLEU/SelfBLEU score (4 if not specified).",
    )
    group.add_argument(
        '--fed',
        nargs='?',
        type=int_in_range(minval=1),
        const=10000,
        help="Number of sample size for FED score.",
    )
    return parser


def save_parser(**kwargs):
    parser = MyArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('save', description="Settings of saving model.")
    group.add_argument(
        '--checkpoint-root', '--ckpt',
        type=path,
        help="Save checkpoint to this directory.",
    )
    if group.get_default('checkpoint_root') is None:  # to avoid interfering SUPPRESS
        group.set_defaults(checkpoint_root=os.getenv('CHECKPOINT_DIR'))

    group.add_argument(
        '--serving-root',
        type=path,
        help='Save serving model to this directory.',
    )
    group.add_argument(
        '--save-period',
        type=int,
        default=1,
        help="Interval (number of epochs) between each saving.",
    )
    return parser


def load_parser(**kwargs):
    parser = MyArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('load', description='Settings of loading saved model.')
    group.add_argument(
        '--model-path',
        type=path,
        required=True,
        help='path of serving model folder.',
    )
    group.add_argument(
        '--version-number',
        default='0',
        help='number of model version.',
    )
    return parser


def develop_parser():
    parser = MyArgumentParser(add_help=False)
    group = parser.add_argument_group('develop', description='Developers only options.')
    group.add_argument(
        '--debug',
        action='store_true',
        help='Whether to print tensorflow warning message.',
    )
    group.add_argument(
        '--profile',
        nargs='?',
        type=path,
        const='./profile_stats',
        help="Export profile stats to file ('./profile_stats' if not specified).",
    )
    return parser


def train_parser():
    parser = MyArgumentParser(add_help=False)
    group = parser.add_argument_group('train', description="Training parameters.")
    group.add_argument(
        '--epochs',
        type=int,
        default=10000,
        help="Number of training epochs.",
    )
    group.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help="Size of data mini-batch.",
    )
    group.add_argument(
        '--random-seed', '--seed',
        type=int,
        help='the global random seed.',
    )
    return parser


def logging_parser():
    parser = MyArgumentParser(add_help=False)
    group = parser.add_argument_group('logging', description="Settings of logging.")
    group.add_argument(
        '--tensorboard',
        nargs='?',
        type=path,
        const=os.getenv('TENSORBOARD_LOGDIR'),
        help="Whether to log experiment on tensorboard "
             "(os.env['TENSORBOARD_LOGDIR'] if not specified).",
    )
    group.add_argument(
        '--tags',
        nargs='+',
        metavar='TAG',
        default=[],
        help="Additional tags for the model (will be used in tensorboard).",
    )
    return parser


def backend_parser():
    parser = MyArgumentParser(add_help=False)
    group = parser.add_argument_group(
        'backend',
        description="Settings of backend graph & session.",
    )
    group.add_argument(
        '--jit',
        action='store_true',
        help='Whether to set global_jit_level = ON_1',
    )
    return parser
