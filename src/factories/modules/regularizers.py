from library.utils import wraps_with_new_signature
from core.objectives.regularizers import LossScaler


def wrap_regularizer(regularizer_cls):

    @wraps_with_new_signature(regularizer_cls)
    def wrapper(coeff, *args, **kwargs):
        return LossScaler(
            regularizer=regularizer_cls(*args, **kwargs),
            coeff=coeff,
        )

    return wrapper
