import tensorflow as tf

from .utils import apply_mask, ComputeOutputMaskMixin1D


class MaskConv1D(ComputeOutputMaskMixin1D, tf.keras.layers.Conv1D):

    def call(self, inputs, mask=None):
        inputs = apply_mask(inputs, mask=mask)
        return super().call(inputs)
