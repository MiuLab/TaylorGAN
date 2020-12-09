import pytest
import tensorflow as tf
from tensorflow.python.keras.layers.recurrent import LSTMCell

from ..recurrent import GRUCell, SkipConnectCells, ZoneoutWrapper


def test_initial_state():
    cell = GRUCell(10)
    x = tf.zeros([3, 4])
    cell(x, cell.get_initial_state(x))


@pytest.mark.parametrize('merge_mode', ['add', 'concat'])
def test_skip_connect_cells(merge_mode):
    cells = [LSTMCell(10) for _ in range(3)]
    skip_cells = SkipConnectCells(cells, merge_mode=merge_mode)

    z = tf.zeros([3, 5])
    state = skip_cells.get_initial_state(z)
    out, new_state = skip_cells(z, state)

    assert len(state) == len(new_state) == len(cells)
    assert len(skip_cells.trainable_variables) == sum([
        len(cell.trainable_variables) for cell in cells])

    if merge_mode == 'add':
        assert out.shape.as_list() == [3, 10]
    elif merge_mode == 'concat':
        assert [cell.kernel.shape[0].value for cell in cells] == [5, 10, 20]
        assert out.shape.as_list() == [3, 10 * len(cells)]


@pytest.mark.parametrize('cell', [
    GRUCell(20),
    LSTMCell(20),
])
def test_zoneout_wrapper(cell, sess):
    wrapped_cell = ZoneoutWrapper(cell, rate=0.5)
    x = tf.ones([3, 5])
    training_place = tf.placeholder(tf.bool, shape=())

    state = wrapped_cell.get_initial_state(x)  # all zero
    assert all(s.shape.as_list() == [3, 20] for s in state)

    out, new_state = wrapped_cell(x, state, training=training_place)
    assert set(wrapped_cell.trainable_variables) == set(cell.trainable_variables)

    sess.run(tf.variables_initializer(wrapped_cell.variables))
    for s, s0 in zip(new_state, state):
        s_val, s0_val = sess.run([s, s0], feed_dict={training_place: True})
        assert (s_val == s0_val).any()  # some units are preserved
        s_val, s0_val = sess.run([s, s0], feed_dict={training_place: False})
        assert (s_val != s0_val).all()  # all units are the interpolation


def test_zoneout_multi_rate():
    wrapped_cell = ZoneoutWrapper(LSTMCell(10), rate=[0.05, 0.5])
    x = tf.ones([3, 5])
    state = wrapped_cell.get_initial_state(x)  # all zero
    wrapped_cell(x, state)
