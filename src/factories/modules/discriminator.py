from functools import partial

import tensorflow as tf
import torch
from torch.nn import Embedding, Sequential

from core.models import Discriminator
from core.objectives.regularizers import (
    SpectralRegularizer,
    EmbeddingRegularizer,
    GradientPenaltyRegularizer,
    WordVectorRegularizer,
)
from flexparse import create_action, FactoryMethod, Namespace
from library.tf_keras_zoo.layers import Dense
from library.tf_keras_zoo.layers.masking import (
    MaskConv1D, MaskAvgPool1D, MaskMaxPool1D, MaskGlobalAvgPool1D,
)
from library.tf_keras_zoo.layers.resnet import ResBlock
from library.torch_zoo.layers import GlobalAvgPool1D

from ..utils import create_factory_action


def create(args: Namespace, meta_data) -> Discriminator:
    (network, info), fix_embeddings = args[MODEL_ARGS]
    print(f"Create discriminator: {info.arg_string}")
    return Discriminator(
        network=network,
        embedder=Embedding.from_pretrained(
            torch.from_numpy(meta_data.load_pretrained_embeddings()),
            freeze=fix_embeddings,
        ),
        name=info.func_name,
    )


tf.keras.utils.get_custom_objects()['lrelu'] = partial(tf.nn.leaky_relu, alpha=0.1)


def cnn(pooling: str = 'avg', padding: str = 'same', activation: str = 'relu'):
    common_kwargs = dict(activation=activation, padding=padding)
    if pooling == 'max':
        PoolingLayer = MaskMaxPool1D
    else:
        PoolingLayer = MaskAvgPool1D

    return Sequential([
        MaskConv1D(filters=512, kernel_size=3, **common_kwargs),
        MaskConv1D(filters=512, kernel_size=4, **common_kwargs),
        PoolingLayer(pool_size=2, padding='same'),
        MaskConv1D(filters=1024, kernel_size=3, **common_kwargs),
        MaskConv1D(filters=1024, kernel_size=4, **common_kwargs),
        MaskGlobalAvgPool1D(),
        Dense(units=1024, activation=activation),
    ])


def resnet(activation: str = 'relu'):
    return Sequential([
        Dense(units=512, activation=activation),
        ResBlock(activation=activation),
        ResBlock(activation=activation),
        ResBlock(activation=activation),
        ResBlock(activation=activation),
        MaskGlobalAvgPool1D(),
        Dense(units=1024, activation=activation),
    ])


MODEL_ARGS = [
    create_factory_action(
        '-d', '--discriminator',
        registry={
            'cnn': cnn,
            'resnet': resnet,
            'test': lambda: GlobalAvgPool1D(dim=1),
        },
        dest='d_network',
        default='cnn(activation=elu)',
        return_info=True,
    ),
    create_action(
        '--d-fix-embeddings',
        action='store_true',
        help="whether to fix embeddings.",
    ),
]

REGULARIZER_ARG = create_factory_action(
    '--d-regularizers',
    registry={
        'spectral': SpectralRegularizer,
        'embedding': EmbeddingRegularizer,
        'grad_penalty': GradientPenaltyRegularizer,
        'word_vec': WordVectorRegularizer,
    },
    nargs='+',
    metavar=f"REGULARIZER(*args{FactoryMethod.COMMA}**kwargs)",
    default=[],
)
