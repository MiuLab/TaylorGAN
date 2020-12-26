import os
import yaml

from dotenv import load_dotenv
from uttut.pipeline.ops import (
    EngTokenizer,
    MergeWhiteSpaceCharacters,
    StripWhiteSpaceCharacters,
    Lowercase,
)

from core.preprocess import UttutPreprocessor
from core.preprocess.adaptors import UttutPipeline
from core.preprocess.config_objects import CorpusConfig, LanguageConfig, Namespace as PathNamespace

from library.my_argparse import MyArgumentParser, SUPPRESS
from library.my_argparse.types import int_in_range
from library.utils import format_id


def preprocess(args, return_meta: bool = False):
    print(f"data_id: {format_id(args.dataset)}")
    print(f"preprocessor_id {format_id('uttut')}")
    corpus_config = CORPUS_CONFIGS[args.dataset]
    preprocessor = UttutPreprocessor(maxlen=args.maxlen, vocab_size=args.vocab_size)
    return preprocessor.preprocess(corpus_config, return_meta=return_meta)


load_dotenv('.env')

CONFIG_PATH = 'datasets/corpus.yaml'
LANGUAGE_CONFIGS = {
    'english': LanguageConfig(
        embedding_path=os.getenv('PRETRAINED_EN_WORD_FASTTEXT_PATH'),
        segmentor=UttutPipeline([
            MergeWhiteSpaceCharacters(),
            StripWhiteSpaceCharacters(),
            Lowercase(),
            EngTokenizer(),
        ]),
        split_token=' ',
    ),
    'test': LanguageConfig(
        embedding_path='datasets/en_fasttext_word2vec_V100D20.msg',
        split_token=' ',
    ),
}


def load_corpus_table(path):
    corpus_table = {}
    with open(path) as f:
        for data_id, corpus_dict in yaml.load(f, Loader=yaml.FullLoader).items():
            config = parse_config(data_id, corpus_dict)
            if config.is_valid():  # TODO else warning?
                corpus_table[data_id] = config

    return corpus_table


def parse_config(data_id, corpus_dict):
    if isinstance(corpus_dict['path'], dict):
        path = PathNamespace(**corpus_dict['path'])
    else:
        path = PathNamespace(train=corpus_dict['path'])

    language_id = corpus_dict['language']
    return CorpusConfig(
        name=data_id,
        path=path,
        language_config=LANGUAGE_CONFIGS[language_id],
        maxlen=corpus_dict.get('maxlen'),
        vocab_size=corpus_dict.get('vocab_size'),
    )


CORPUS_CONFIGS = load_corpus_table(CONFIG_PATH)


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
