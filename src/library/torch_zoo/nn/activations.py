from torch.nn import (  # noqa: F401
    Module,
    ELU,
    LeakyReLU,
    ReLU,
    GELU,
    SELU,
)


_ACTIVATION_CLS = {
    name.lower(): module_cls
    for name, module_cls in locals().items()
    if isinstance(module_cls, type) and issubclass(module_cls, Module) and module_cls != Module
}


class LiteralHint:

    def __init__(self, keys, ellipsis: bool = False):
        self.keys = list(keys)
        self.ellipsis = ellipsis

    def __repr__(self):
        if self.ellipsis and len(self.keys) >= 3:
            return f"<{self.keys[0]!r}|...|{self.keys[-1]!r}>"
        else:
            return f"<{'|'.join(map(repr, self.keys))}>"


TYPE_HINT = LiteralHint(_ACTIVATION_CLS.keys())


def deserialize(key):
    return _ACTIVATION_CLS[key]
