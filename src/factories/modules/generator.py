from functools import partial

import torch
from torch.nn import Embedding, GRUCell, Linear, Sequential

from core.models import Generator, AutoRegressiveGenerator
from core.objectives.regularizers import (
    LossScaler,
    SpectralRegularizer,
    EmbeddingRegularizer,
    EntropyRegularizer,
)
from flexparse import create_action, Namespace, LookUpCall
from library.utils import NamedObject

from ..utils import create_factory_action


def create(args: Namespace, meta_data) -> Generator:
    cell_func, fix_embeddings, tie_embeddings = args[MODEL_ARGS]
    print(f"Create generator: {cell_func.argument_info.arg_string}")

    embedding_matrix = torch.from_numpy(meta_data.load_pretrained_embeddings())
    embedder = Embedding.from_pretrained(embedding_matrix, freeze=fix_embeddings)
    presoftmax_layer = Linear(embedder.embedding_dim, embedder.num_embeddings)
    if tie_embeddings:
        presoftmax_layer.weight = embedder.weight
    else:
        presoftmax_layer.weight.data.copy_(embedder.weight)

    cell = cell_func(input_size=embedder.embedding_dim)
    return NamedObject(
        AutoRegressiveGenerator(
            cell=cell,
            embedder=embedder,
            output_layer=Sequential(
                Linear(cell.hidden_size, embedder.embedding_dim, bias=False),
                presoftmax_layer,
            ),
            special_token_config=meta_data.special_token_config,
        ),
        name=cell_func.argument_info.func_name,
    )


def gru_cell(units: int = 1024):
    return partial(GRUCell, hidden_size=units)


MODEL_ARGS = [
    create_factory_action(
        '-g', '--generator',
        type=LookUpCall(
            {
                'gru': gru_cell,
                'test': lambda: partial(GRUCell, hidden_size=10),
            },
            set_info=True,
        ),
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
    type=LookUpCall({
        'spectral': LossScaler.as_constructor(SpectralRegularizer),
        'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
        'entropy': LossScaler.as_constructor(EntropyRegularizer),
    }),
    nargs='+',
    metavar="REGULARIZER(*args, **kwargs)",
    default=[],
)
