from itertools import chain
from typing import List

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class GRUCell(tf.keras.layers.GRUCell):

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [super().get_initial_state(inputs, batch_size, dtype)]


class SkipConnectCells(tf.keras.layers.Layer):

    def __init__(self, cells: List[tf.keras.layers.Layer], merge_mode='concat'):
        super().__init__()
        self.cells = cells
        if merge_mode not in ('add', 'concat'):
            raise KeyError("`merge_mode` should be `add` or `concat`!")
        self.merge_mode = merge_mode

    def call(self, inputs, state):
        new_states = []
        merged_output = None
        for cell, s in zip(self.cells, state):
            output, new_s = cell(inputs, s)
            if merged_output is None:
                merged_output = output
            else:
                merged_output = self._merge_output(output, merged_output)  # skip connection
            inputs = merged_output
            new_states.append(new_s)

        return merged_output, new_states

    def get_initial_state(self, *args, **kwargs):
        return [cell.get_initial_state(*args, **kwargs) for cell in self.cells]

    def _merge_output(self, new_output, old_output):
        if self.merge_mode == 'add':
            return old_output + new_output
        elif self.merge_mode == 'concat':
            return tf.concat([old_output, new_output], axis=1)
        else:
            raise AssertionError("Invalid `merge_mode` should be raised when init!")

    @property
    def trainable_weights(self):
        return list(chain.from_iterable([cell.trainable_weights for cell in self.cells]))

    @property
    def non_trainable_weights(self):
        return list(chain.from_iterable([cell.non_trainable_weights for cell in self.cells]))

    @property
    def updates(self):
        return list(chain.from_iterable([cell.updates for cell in self.cells]))


class ZoneoutWrapper(tf.keras.layers.Layer):
    """Reference: https://arxiv.org/pdf/1606.01305.pdf
    """

    def __init__(self, cell, rate):
        super().__init__()
        if hasattr(cell, 'cells'):
            raise ValueError("Can't support `StackedRNNCells`!")
        if isinstance(rate, (list, tuple)):
            if len(rate) != len(_tolist(cell.state_size)):
                raise ValueError(
                    "`rate` should be float or list with same length as `cell.state_size`!",
                )

        self.cell = cell
        self.rate = rate

    def get_initial_state(self, *args, **kwargs):
        return self.cell.get_initial_state(*args, **kwargs)

    def call(self, inputs, state, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if isinstance(self.rate, (list, tuple)):
            rate_list = self.rate
        else:
            rate_list = [self.rate for _ in state]

        out, new_state = self.cell(inputs, state)
        residual = [new_s - s for new_s, s in zip(new_state, state)]
        zoneout_state = [
            s + self._scaled_dropout(res, training, rate)
            for s, res, rate in zip(state, residual, rate_list)
        ]
        return out, zoneout_state

    def _scaled_dropout(self, x, training, rate):
        output = tf_utils.smart_cond(
            training,
            lambda: tf.nn.dropout(x, rate=rate),
            lambda: tf.identity(x),
        ) * (1 - rate)  # cancel the scaling made by tf.nn.dropout
        return output


def _tolist(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]
