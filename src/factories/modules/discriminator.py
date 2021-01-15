import torch
from torch.nn import Embedding, Linear

from core.models import Discriminator
from core.objectives.regularizers import (
    LossScaler,
    SpectralRegularizer,
    EmbeddingRegularizer,
    GradientPenaltyRegularizer,
    WordVectorRegularizer,
)
from flexparse import create_action, Namespace, LookUpCall
from library.torch_zoo.nn import activations, LambdaModule
from library.torch_zoo.nn.resnet import ResBlock
from library.torch_zoo.nn.masking import (
    MaskConv1d, MaskAvgPool1d, MaskGlobalAvgPool1d, MaskSequential,
)
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


def cnn(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        MaskConv1d(input_size, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskConv1d(512, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskAvgPool1d(kernel_size=2),
        MaskConv1d(512, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskConv1d(1024, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskGlobalAvgPool1d(),
        Linear(1024, 1024),
        ActivationLayer(),
    )


def resnet(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        Linear(input_size, 512),
        ActivationLayer(),
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        MaskGlobalAvgPool1d(),
        Linear(512, 512),
        ActivationLayer(),
    )


MODEL_ARGS = [
    create_factory_action(
        '-d', '--discriminator',
        type=LookUpCall(
            {
                key: ArgumentBinder(func, preserved=['input_size'])
                for key, func in [
                    ('cnn', cnn),
                    ('resnet', resnet),
                    ('test', lambda input_size: MaskGlobalAvgPool1d(dim=1)),
                ]
            },
            set_info=True,
        ),
        default="cnn(activation='elu')",
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
