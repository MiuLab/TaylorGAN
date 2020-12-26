import tensorflow as tf

from core.models import Generator, AutoRegressiveGenerator
from core.objectives.regularizers import (
    SpectralRegularizer,
    EmbeddingRegularizer,
    EntropyRegularizer,
)
from flexparse import create_action, FactoryMethod, Namespace
from library.tf_keras_zoo.layers import Dense, Embedding, GRUCell, StackedRNNCells
from library.tf_keras_zoo.layers.embeddings import OutputEmbedding
from library.tf_keras_zoo.layers.recurrent import SkipConnectCells
from library.tf_keras_zoo.networks import Sequential

from ..utils import create_factory_action


def create(args: Namespace, meta_data) -> Generator:
    (cell, info), fix_embeddings, tie_embeddings = args[MODEL_ARGS]
    print(f"Create generator: {info.arg_string}")

    embedding_matrix = meta_data.load_pretrained_embeddings()
    embedder = Embedding.from_weights(embedding_matrix, trainable=not fix_embeddings)
    if tie_embeddings:
        presoftmax_layer = OutputEmbedding(embedder, use_bias=True, name='pre_softmax')
    else:
        presoftmax_layer = Dense(
            embedder.vocab_size,
            kernel_initializer=tf.constant_initializer(embedding_matrix.T),
            use_bias=True,
            name='pre_softmax',
        )

    return AutoRegressiveGenerator(
        cell=cell,
        embedder=embedder,
        output_layer=Sequential([
            Dense(units=embedder.total_dim, name='projection', use_bias=False),
            presoftmax_layer,
        ]),
        special_token_config=meta_data.special_token_config,
        name=info.func_name,
    )


def gru_cell(units: int = 1024, layers: int = 1, merge_mode: str = None):
    cells = [
        GRUCell(
            units=units,
            recurrent_activation='sigmoid',
            reset_after=True,
            implementation=2,
        )
        for _ in range(layers)
    ]
    if len(cells) == 1:
        return cells[0]
    elif merge_mode:
        return SkipConnectCells(cells, merge_mode=merge_mode)
    else:
        return StackedRNNCells(cells)


MODEL_ARGS = [
    create_factory_action(
        '-g', '--generator',
        dest='g_cell',
        registry={'gru': gru_cell, 'test': lambda: GRUCell(units=10)},
        return_info=True,
        default='gru',
    ),
    create_action(
        '--tie-embeddings',
        action='store_true',
        help="whether to tie the weights of generator's input/presoftmax embeddings.",
    ),
    create_action(
        '--g-fix-embeddings',
        action='store_true',
        help="whether to fix embeddings.",
    ),
]

REGULARIZER_ARG = create_factory_action(
    '--g-regularizers',
    registry={
        'spectral': SpectralRegularizer,
        'embedding': EmbeddingRegularizer,
        'entropy': EntropyRegularizer,
    },
    nargs='+',
    metavar=f"REGULARIZER(*args{FactoryMethod.COMMA}**kwargs)",
    default=[],
)
