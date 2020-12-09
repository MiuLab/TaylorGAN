from library.my_argparse import MyArgumentParser, SUPPRESS
from library.my_argparse.types import int_in_range
from library.utils import format_id

from core.preprocess import UttutPreprocessor
from datasets import CORPUS_CONFIGS


def preprocess(args, return_meta: bool = False):
    print(f"data_id: {format_id(args.dataset)}")
    print(f"preprocessor_id {format_id('uttut')}")
    corpus_config = CORPUS_CONFIGS[args.dataset]
    preprocessor = UttutPreprocessor(maxlen=args.maxlen, vocab_size=args.vocab_size)
    return preprocessor.preprocess(corpus_config, return_meta=return_meta)


def create_parser(**kwargs):
    parser = MyArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group(
        'data',
        description="Data corpus and preprocessing configurations.",
    )
    group.add_argument(
        '--dataset',
        choices=CORPUS_CONFIGS.keys(),
        required=True,
        default=SUPPRESS,
        help='the choice of corpus.',
    )
    group.add_argument(
        '--maxlen',
        type=int_in_range(minval=1),
        help="the max length of sequence padding. "
             "(use the value declared in corpus_config if not given)",
    )
    group.add_argument(
        '--vocab_size',
        type=int_in_range(minval=1),
        help="the maximum number of tokens. ordered by descending frequency. "
             "(use the value declared in corpus_config if not given)",
    )
    return parser
