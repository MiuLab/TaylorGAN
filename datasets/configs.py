import os
import yaml

from dotenv import load_dotenv
from uttut.pipeline.ops import (
    EngTokenizer,
    MergeWhiteSpaceCharacters,
    StripWhiteSpaceCharacters,
    Lowercase,
)

from core.preprocess.adaptors import UttutPipeline
from core.preprocess.config_objects import CorpusConfig, LanguageConfig, Namespace


load_dotenv('.env')
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
        path = Namespace(**corpus_dict['path'])
    else:
        path = Namespace(train=corpus_dict['path'])

    language_id = corpus_dict['language']
    return CorpusConfig(
        name=data_id,
        path=path,
        language_config=LANGUAGE_CONFIGS[language_id],
        maxlen=corpus_dict.get('maxlen'),
        vocab_size=corpus_dict.get('vocab_size'),
    )


CORPUS_CONFIGS = load_corpus_table('datasets/corpus.yaml')
