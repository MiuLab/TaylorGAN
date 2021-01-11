from functools import partial

import torch
from torch.nn import AvgPool1d, Conv1d, Embedding, Linear, ReLU, Sequential

from core.models import Discriminator
from core.objectives.regularizers import (
    LossScaler,
    SpectralRegularizer,
    EmbeddingRegularizer,
    GradientPenaltyRegularizer,
    WordVectorRegularizer,
)
from flexparse import create_action, Namespace, LookUpCall
# from library.tf_keras_zoo.layers.masking import (
#     MaskConv1D, MaskAvgPool1D, MaskMaxPool1D, MaskGlobalAvgPool1D,
# )
# from library.tf_keras_zoo.layers.resnet import ResBlock
from library.torch_zoo.layers import GlobalAvgPool1D, LambdaModule
from library.utils import ArgumentBinder, NamedObject

from ..utils import create_factory_action


def create(args: Namespace, meta_data) -> Discriminator:
    network_func, fix_embeddings = args[MODEL_ARGS]
    print(f"Create discriminator: {network_func.argument_info.arg_string}")
    embedder = Embedding.from_pretrained(
        torch.from_numpy(meta_data.load_pretrained_embeddings()),
        freeze=fix_embeddings,
    )

    return NamedObject(
        Discriminator(
            network=network_func(embedder.embedding_dim),
            embedder=embedder,
        ),
        name=network_func.argument_info.func_name,
    )


def cnn(input_size):
    PoolingLayer = AvgPool1d
    ActivationLayer = ReLU
    return Sequential(
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        Conv1d(input_size, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        Conv1d(512, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        PoolingLayer(kernel_size=2),
        Conv1d(512, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        Conv1d(1024, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        GlobalAvgPool1D(dim=2),
        Linear(1024, 1024),
        ActivationLayer(),
    )


# def resnet(activation: str = 'relu'):
#     return Sequential([
#         Dense(units=512, activation=activation),
#         ResBlock(activation=activation),
#         ResBlock(activation=activation),
#         ResBlock(activation=activation),
#         ResBlock(activation=activation),
#         MaskGlobalAvgPool1D(),
#         Dense(units=1024, activation=activation),
#     ])


MODEL_ARGS = [
    create_factory_action(
        '-d', '--discriminator',
        type=LookUpCall(
            {
                key: ArgumentBinder(func, preserved=['input_size'])
                for key, func in [
                    ('cnn', cnn),
                    ('test', lambda input_size: GlobalAvgPool1D(dim=1)),
                ]
            },
            set_info=True,
        ),
        default='cnn(activation=elu)',
    ),
    create_action(
        '--d-fix-embeddings',
        action='store_true',
        help="whether to fix embeddings.",
    ),
]

REGULARIZER_ARG = create_factory_action(
    '--d-regularizers',
    type=LookUpCall({
        'spectral': LossScaler.as_constructor(SpectralRegularizer),
        'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
        'grad_penalty': LossScaler.as_constructor(GradientPenaltyRegularizer),
        'word_vec': LossScaler.as_constructor(WordVectorRegularizer),
    }),
    nargs='+',
    metavar="REGULARIZER(*args, **kwargs)",
    default=[],
)
