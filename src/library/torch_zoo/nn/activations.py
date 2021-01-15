from torch.nn import (  # noqa: F401
    Module,
    ELU,
    LeakyReLU,
    ReLU,
    ReLU6,
    GELU,
    Softplus,
    SELU,
)


_ACTIVATION_CLS = {
    name.lower(): module_cls
    for name, module_cls in locals().items()
    if isinstance(module_cls, type) and issubclass(module_cls, Module) and module_cls != Module
}


def deserialize(key):
    return _ACTIVATION_CLS[key]
