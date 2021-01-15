from torch.nn import ReLU, ELU, LeakyReLU

from ..activations import deserialize


def test_deserialize():
    assert deserialize('relu') == ReLU
    assert deserialize('elu') == ELU
    assert deserialize('leakyrelu') == LeakyReLU
