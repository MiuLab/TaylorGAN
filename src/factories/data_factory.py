import os
import yaml

from dotenv import load_dotenv
from flexparse import (
    SUPPRESS, create_action, IntRange, LookUp, Namespace,
    ArgumentParser,
)
from uttut.pipeline.ops import (
    EngTokenizer,
    MergeWhiteSpaceCharacters,
    StripWhiteSpaceCharacters,
    Lowercase,
)

from core.preprocess import UttutPreprocessor
from core.preprocess.adaptors import UttutPipeline
from core.preprocess.config_objects import CorpusConfig, LanguageConfig, Namespace as PathNamespace
from library.utils import format_id, format_path, NamedDict


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


def preprocess(args: Namespace, return_meta: bool = False):
    dataset, maxlen, vocab_size = args[ARGS]
    print(f"data_id: {format_id(dataset)}")
    print(f"preprocessor_id {format_id('uttut')}")
    preprocessor = UttutPreprocessor(maxlen=maxlen, vocab_size=vocab_size)
    return preprocessor.preprocess(dataset, return_meta=return_meta)


def load_corpus_table(path):
    corpus_table = NamedDict()
    with open(path) as f:
        for data_id, corpus_dict in yaml.load(f, Loader=yaml.FullLoader).items():
            config = parse_config(corpus_dict)
            if config.is_valid():  # TODO else warning?
                corpus_table[data_id] = config

    return corpus_table


def parse_config(corpus_dict):
    if isinstance(corpus_dict['path'], dict):
        path = PathNamespace(**corpus_dict['path'])
    else:
        path = PathNamespace(train=corpus_dict['path'])

    language_id = corpus_dict['language']
    return CorpusConfig(
        path=path,
        language_config=LANGUAGE_CONFIGS[language_id],
        maxlen=corpus_dict.get('maxlen'),
        vocab_size=corpus_dict.get('vocab_size'),
    )


ARGS = [
    create_action(
        '--dataset',
        type=LookUp(load_corpus_table(CONFIG_PATH)),
        required=True,
        default=SUPPRESS,
        help='the choice of corpus.',
    ),
    create_action(
        '--maxlen',
        type=IntRange(minval=1),
        help="the max length of sequence padding. "
             f"(use the value declared in {format_path(CONFIG_PATH)} if not given)",
    ),
    create_action(
        '--vocab-size',
        type=IntRange(minval=1),
        help="the maximum number of tokens. ordered by descending frequency. "
             f"(use the value declared in {format_path(CONFIG_PATH)} if not given)",
    ),
]

PARSER = ArgumentParser(add_help=False)
PARSER.add_argument_group(
    'data',
    description="data corpus and preprocessing configurations.",
    actions=ARGS,
)
