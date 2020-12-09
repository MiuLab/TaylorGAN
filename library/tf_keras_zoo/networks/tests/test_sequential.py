import pytest

import tensorflow as tf

from ..sequential import Sequential


@pytest.fixture
def sequential():
    return Sequential([
        tf.keras.layers.Embedding(20, 10),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.Dense(5),
        tf.keras.layers.MaxPooling1D(),
    ])


@pytest.fixture(scope='module')
def int_inputs():
    return tf.zeros([2, 3], dtype=tf.int32)


@pytest.fixture(scope='module')
def float_inputs():
    return tf.ones([5, 4, 3], dtype=tf.float32)


def test_empty_sequential(int_inputs, float_inputs):
    seq = Sequential()
    assert seq(int_inputs) is int_inputs
    assert seq(float_inputs) is float_inputs


def test_build_sublayers_when_first_called(int_inputs, sequential):
    assert all(not layer.built for layer in sequential.layers)
    sequential(int_inputs)
    assert all(layer.built for layer in sequential.layers)


def test_context_manager_work_when_first_called(sequential):
    new_graph = tf.Graph()
    assert new_graph is not tf.get_default_graph()
    assert len(sequential.variables) == 0

    with new_graph.as_default(), tf.variable_scope('scope'):
        int_inputs = tf.zeros([2, 3], dtype=tf.int32)
        sequential(int_inputs)

    variables = sequential.variables
    assert len(variables) > 0
    assert all(var.graph is new_graph for var in variables)
    assert all(var.name.startswith('scope') for var in variables)


def test_mask_computed_by_layer_propagate(float_inputs):
    sequential_with_masking = Sequential([
        tf.keras.layers.Masking(),
        tf.keras.layers.Dense(5),
    ])
    assert all(layer.supports_masking for layer in sequential_with_masking.layers)
    outputs = sequential_with_masking(float_inputs)
    assert outputs._keras_mask is not None


def test_mask_given_in_inputs_propagate(float_inputs):
    sequential_supports_mask = Sequential([
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(5),
    ])
    assert all(layer.supports_masking for layer in sequential_supports_mask.layers)
    mask = tf.ones([5, 4], dtype=tf.bool)
    outputs = sequential_supports_mask(
        float_inputs,
        mask=mask,
    )
    assert outputs._keras_mask is mask  # since Dense simply pass it.


def test_masked_inputs_propagate(float_inputs):
    masked_inputs = tf.keras.layers.Masking()(float_inputs)
    sequential_supports_mask = Sequential([
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(5),
    ])
    assert all(layer.supports_masking for layer in sequential_supports_mask.layers)
    outputs = sequential_supports_mask(masked_inputs)
    assert outputs._keras_mask is masked_inputs._keras_mask
