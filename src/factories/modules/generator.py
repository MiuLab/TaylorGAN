import tensorflow as tf

from library.my_argparse.actions import IdKwargs
from library.tf_keras_zoo.layers import Dense, Embedding, GRUCell, StackedRNNCells
from library.tf_keras_zoo.layers.embeddings import OutputEmbedding
from library.tf_keras_zoo.layers.recurrent import SkipConnectCells
from library.tf_keras_zoo.networks import Sequential
from library.utils import format_id

from core.models import Generator, AutoRegressiveGenerator
from core.objectives.regularizers import (
    SpectralRegularizer,
    EmbeddingRegularizer,
    EntropyRegularizer,
)
from factories.base import SimpleFactory, get_help_of_id_kwargs
from .base import ModuleFactory


class GeneratorSimpleFactory(SimpleFactory):

    def create(self, args, meta_data) -> Generator:
        embedding_matrix = meta_data.load_pretrained_embeddings()
        embedder = Embedding.from_weights(embedding_matrix, trainable=not args.g_fix_embeddings)
        if args.tie_embeddings:
            presoftmax_layer = OutputEmbedding(embedder, use_bias=True, name='pre_softmax')
        else:
            presoftmax_layer = Dense(
                embedder.vocab_size,
                kernel_initializer=tf.constant_initializer(embedding_matrix.T),
                use_bias=True,
                name='pre_softmax',
            )

        generator_id, kwargs = args.generator
        print(f"Create generator: {format_id(generator_id)}")
        return AutoRegressiveGenerator(
            cell=self.table[generator_id](**kwargs),
            embedder=embedder,
            output_layer=Sequential([
                Dense(units=embedder.total_dim, name='projection', use_bias=False),
                presoftmax_layer,
            ]),
            special_token_config=meta_data.special_token_config,
            name=generator_id,
        )

    def add_argument_to(self, holder):
        holder.add_argument(
            '-g', '--generator',
            action=IdKwargs,
            id_choices=self.table.keys(),
            default=IdKwargs.IdKwargsPair('gru'),
            split_token=',',
            metavar='GENERATOR_ID',
            help=f"the type of generator.\n{get_help_of_id_kwargs(self.table)}\n",
        )
        holder.add_argument(
            '--tie-embeddings',
            action='store_true',
            help="whether to tie the weights of generator's input/presoftmax embeddings.",
        )
        holder.add_argument(
            '--g-fix-embeddings',
            action='store_true',
            help="whether to fix embeddings.",
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


generator_factory = ModuleFactory(
    module_factory=GeneratorSimpleFactory({
        'gru': gru_cell,
        'test': lambda: GRUCell(units=10),
    }),
    module_name=Generator.scope.lower(),
)

generator_factory.regularizer_table.update(dict(
    spectral=SpectralRegularizer,
    embedding=EmbeddingRegularizer,
    entropy=EntropyRegularizer,
))
