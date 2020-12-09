import pytest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from library.tf_keras_zoo.networks import Sequential
from ..embeddings import Embedding, OutputEmbedding


@pytest.fixture(scope='module')
def inputs():
    return tf.placeholder(tf.int32, shape=[None, 3])


@pytest.fixture(scope='module')
def mask(inputs):
    return tf.placeholder(tf.bool, shape=inputs.shape)


def test_output_shape(inputs):
    embed_layer = Embedding(vocab_size=3, embeddings_dim=5, mask_index=1)
    outputs = embed_layer(inputs)
    maxlen = inputs.shape[1].value
    assert outputs.shape.as_list() == [None, maxlen, 5]


@pytest.mark.parametrize('mask_index', [
    1,
    [1, 2, 3],
])
def test_mask_index(inputs, mask, sess, mask_index):
    maxlen = inputs.shape[1].value
    embed_layer = Embedding(vocab_size=10, embeddings_dim=5, mask_index=mask_index)
    seq = Sequential([embed_layer])  # to handle mask propagate
    outputs = seq(inputs, mask=mask)

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    input_val = np.random.randint(0, embed_layer.vocab_size, size=[5, maxlen])
    mask_val = np.random.randint(0, 2, size=[5, maxlen], dtype=np.bool)
    output_mask_val = sess.run(
        outputs._keras_mask,
        feed_dict={inputs: input_val, mask: mask_val},
    )
    if isinstance(mask_index, int):
        expected_val = input_val != mask_index
    else:
        expected_val = np.isin(input_val, mask_index, invert=True)

    expected_val = np.logical_and(mask_val, expected_val)

    assert np.array_equal(output_mask_val, expected_val)


@pytest.mark.parametrize('invalid_mask_index', [
    'mask',  # not int
    3.,  # not int
    ['mask'],  # not int
    6,  # > vocab_size
    [1, 2, [3]],  # nested
    -1,
])
def test_init_from_invalid_mask_index_raise(invalid_mask_index):
    with pytest.raises(ValueError):
        Embedding(vocab_size=5, embeddings_dim=5, mask_index=invalid_mask_index)


def test_dropout(inputs, sess):
    embed_layer = Embedding(
        vocab_size=10, embeddings_dim=5, dropout=0.8, embeddings_initializer='ones')
    training = tf.placeholder(dtype=tf.bool, shape=())
    outputs = embed_layer(inputs, mask=mask, training=training)

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    maxlen = inputs.shape[1].value
    input_val = np.random.randint(0, embed_layer.vocab_size, size=[5, maxlen])
    dropped_out = sess.run(
        outputs,
        feed_dict={inputs: input_val, training: True},
    )
    assert np.all(dropped_out == 0., axis=2).any()  # on embedding dims

    no_dropped_out = sess.run(
        outputs,
        feed_dict={inputs: input_val, training: False},
    )
    assert (no_dropped_out != 0.).all()


@pytest.mark.parametrize('constant', [False, True])
def test_construct_from_weights(inputs, sess, constant):
    weights = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    embed_layer = Embedding.from_weights(weights, constant=constant)
    embed_layer(inputs)  # to build variables

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))
    weights_val = sess.run(embed_layer.embeddings)

    np.testing.assert_array_almost_equal(weights_val, weights)


@pytest.mark.parametrize('constant', [False, True])
@pytest.mark.parametrize('auxiliary_tokens, extend_dims', [
    (0, 2),
    (2, 0),
    (2, 2),
])
def test_extend_partially_trainable(inputs, sess, constant, auxiliary_tokens, extend_dims):
    maxlen = inputs.shape[1].value
    vocab_size, embeddings_dim = 5, 3
    embed_layer = Embedding.from_weights(
        np.random.uniform(size=[vocab_size, embeddings_dim]).astype(np.float32),
        constant=constant,
        trainable=False,
        auxiliary_tokens=auxiliary_tokens,
        extend_dims=extend_dims,
    )
    word_vec = embed_layer(inputs)

    len_trainable_variables = (1 if auxiliary_tokens else 0) + (1 if extend_dims else 0)
    len_non_trainable_variables = 0 if constant else 1

    assert len(embed_layer.trainable_variables) == len_trainable_variables
    assert len(embed_layer.non_trainable_variables) == len_non_trainable_variables
    assert len(embed_layer.variables) == len_trainable_variables + len_non_trainable_variables
    assert embed_layer.total_embeddings.shape.as_list() == [
        vocab_size + auxiliary_tokens,
        embeddings_dim + extend_dims,
    ]

    update_op = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_sum(word_vec))

    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    original_weights_val = sess.run(embed_layer.total_embeddings)
    sess.run(
        update_op,
        feed_dict={inputs: np.random.choice(
            vocab_size + auxiliary_tokens,
            size=[10, maxlen],
        )},
    )
    new_weights_val = sess.run(embed_layer.total_embeddings)

    # after update:
    # original part should keep
    np.testing.assert_array_almost_equal(
        original_weights_val[: vocab_size, : embeddings_dim],
        new_weights_val[: vocab_size, : embeddings_dim],
    )
    # others (auxiliary tokens) should change.
    if auxiliary_tokens:
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(
                original_weights_val[vocab_size:],
                new_weights_val[vocab_size:],
            )
    if extend_dims:
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(
                original_weights_val[:, embeddings_dim:],
                new_weights_val[:, embeddings_dim:],
            )


@pytest.mark.parametrize('invalid_weights', [
    np.zeros([5]),
    np.zeros([1, 2, 3]),
])
def test_construct_from_invalid_weights_raise(invalid_weights):
    with pytest.raises(ValueError):
        Embedding.from_weights(invalid_weights)


@pytest.mark.parametrize('constant', [True, False])
@pytest.mark.parametrize('auxiliary_tokens', [0, 2])
@pytest.mark.parametrize('extend_dims', [0, 5])
def test_freeze_success(inputs, sess, constant, auxiliary_tokens, extend_dims):
    # build graph with constant embedding layer
    embed_layer = Embedding.from_weights(
        np.random.rand(5, 10).astype(np.float32),
        constant=constant,
        auxiliary_tokens=auxiliary_tokens,
        extend_dims=extend_dims,
    )
    outputs = embed_layer(inputs)
    sess.run(tf.variables_initializer(var_list=embed_layer.variables))

    # freeze graph
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=[outputs.op.name],  # node name == op name
    )

    # check frozen graph have the same embedding as before
    new_sess = create_session_from_graphdef(frozen_graph_def)
    new_inputs = new_sess.graph.get_tensor_by_name(inputs.name)
    new_outputs = new_sess.graph.get_tensor_by_name(outputs.name)

    maxlen = inputs.shape[1].value
    inputs_val = np.random.choice(embed_layer.vocab_size, size=[5, maxlen])
    outputs_val = sess.run(outputs, feed_dict={inputs: inputs_val})
    new_outputs_val = new_sess.run(new_outputs, feed_dict={new_inputs: inputs_val})

    np.testing.assert_array_almost_equal(outputs_val, new_outputs_val)


def create_session_from_graphdef(graph_def):
    """
    Create new session from given tf.GraphDef object
    Arg:
       graph_def (tf.GraphDef): a tf.GraphDef object
    Return:
       session (tf.Session): a new session with given graph_def
    """
    new_sess = tf.Session(graph=tf.Graph())
    with new_sess.graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return new_sess


@pytest.mark.parametrize('embedder', [
    Embedding(vocab_size=10, embeddings_dim=20),
    Embedding.from_weights(np.random.rand(10, 20), constant=True),
    Embedding.from_weights(np.random.rand(10, 20), extend_dims=10),
    Embedding.from_weights(np.random.rand(10, 20), constant=True, extend_dims=10),
])
def test_tie_embedding(embedder):
    dense = OutputEmbedding(embedder, use_bias=True)

    x = tf.zeros(shape=[1, 10], dtype=tf.int32)
    word_vector = embedder(x)
    logits = dense(word_vector)
    assert logits.shape[-1].value == embedder.vocab_size
    assert set(dense.variables) == {*embedder.variables, dense.bias}
    assert set(dense.trainable_variables) == {*embedder.trainable_variables, dense.bias}
