from functools import partial

import tensorflow as tf

from library.my_argparse.actions import IdKwargs
from library.tf_keras_zoo.layers import Dense, Embedding
from library.tf_keras_zoo.layers.masking import (
    MaskConv1D, MaskAvgPool1D, MaskMaxPool1D, MaskGlobalAvgPool1D,
)
from library.tf_keras_zoo.layers.resnet import ResBlock
from library.tf_keras_zoo.networks import Sequential
from library.utils import format_id

from core.models import Discriminator
from core.objectives.regularizers import (
    SpectralRegularizer,
    EmbeddingRegularizer,
    GradientPenaltyRegularizer,
    WordVectorRegularizer,
)
from factories.base import SimpleFactory, get_help_of_id_kwargs
from .base import ModuleFactory


class DiscriminatorSimpleFactory(SimpleFactory):

    def create(self, args, meta_data) -> Discriminator:
        discriminator_id, kwargs = args.discriminator
        print(f"Create discriminator: {format_id(discriminator_id)}")
        return Discriminator(
            network=self.table[discriminator_id](**kwargs),
            embedder=Embedding.from_weights(
                weights=meta_data.load_pretrained_embeddings(),
                trainable=not args.d_fix_embeddings,
            ),
            name=discriminator_id,
        )

    def add_argument_to(self, holder):
        holder.add_argument(
            '-d', '--discriminator',
            action=IdKwargs,
            id_choices=self.table.keys(),
            default=IdKwargs.IdKwargsPair('cnn', activation='elu'),
            metavar='DISCRIMINATOR_ID',
            split_token=',',
            help=f"the type of discriminator.\n{get_help_of_id_kwargs(self.table)}\n",
        )
        holder.add_argument(
            '--d-fix-embeddings',
            action='store_true',
            help="whether to fix embeddings.",
        )


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


tf.keras.utils.get_custom_objects()['lrelu'] = partial(tf.nn.leaky_relu, alpha=0.1)
discriminator_factory = ModuleFactory(
    module_factory=DiscriminatorSimpleFactory({
        'cnn': cnn,
        'resnet': resnet,
        'test': lambda: Sequential([Dense(10), MaskGlobalAvgPool1D()]),
    }),
    module_name=Discriminator.scope.lower(),
)

discriminator_factory.regularizer_table.update(dict(
    spectral=SpectralRegularizer,
    embedding=EmbeddingRegularizer,
    grad_penalty=GradientPenaltyRegularizer,
    word_vec=WordVectorRegularizer,
))
