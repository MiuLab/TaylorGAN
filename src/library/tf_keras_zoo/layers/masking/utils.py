import re

import tensorflow as tf

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops


_APPLY_MASK_MUL = 'apply_mask_mul'


def apply_mask(inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    if mask is None:
        return inputs
    if re.match(f".*{_APPLY_MASK_MUL}(_[0-9]+)?", inputs.op.name):  # e.g. .../apply_mask_mul_1
        return inputs  # mask has already been applied

    i_rank = inputs.shape.ndims
    m_rank = mask.shape.ndims
    mask = tf.cast(mask, inputs.dtype)
    if i_rank > m_rank:
        indices = tuple(
            slice(None) if d < m_rank else tf.newaxis
            for d in range(i_rank)
        )  # expand multiple dims
        mask = mask[indices]
    elif i_rank < m_rank:
        raise ValueError(f"Invalid mask rank > inputs rank! {m_rank} > {i_rank}")

    return tf.multiply(inputs, mask, name=_APPLY_MASK_MUL)


class ComputeOutputMaskMixin1D:

    def __init__(self, mask_threshold: int = None, **kwargs):
        super().__init__(**kwargs)
        size = self._get_window_size()
        if not (mask_threshold is None or 1 <= mask_threshold <= size):
            raise ValueError(f"`mask_threshold` should be in [1, {size}]")
        self.mask_threshold = mask_threshold
        self.supports_masking = True

    def build(self, input_shape):
        super().build(input_shape)  # super of host class
        if self.mask_threshold:
            # just for given threshold
            size = self._get_window_size()
            self.mask_kernel = tf.ones((size, 1, 1), dtype=self.dtype)
            if self.padding == 'causal':
                op_padding = 'valid'
            else:
                op_padding = self.padding

            mask_shape = (*input_shape.as_list()[:-1], 1)
            self._mask_op = nn_ops.Convolution(
                tf.TensorShape(mask_shape),
                filter_shape=self.mask_kernel.get_shape(),
                strides=self.strides,
                padding=op_padding.upper(),
                data_format=conv_utils.convert_data_format(self.data_format, 3),
            )

    def _get_window_size(self):
        for size_attr in ('kernel_size', 'pool_size'):
            if hasattr(self, size_attr):
                return getattr(self, size_attr)[0]

        raise AttributeError(
            f"Host class of {self.__class__.__name__} should have either "
            "`kernel_size` or `pool_size`.",
        )

    def compute_mask(self, inputs, mask):
        if mask is None:
            return None

        if self.mask_threshold:
            return self._compute_mask_with_threshold(mask)

        # assume mask is continuoul: e.g sequence
        size = self._get_window_size()
        strides = self.strides[0]
        if self.padding == 'valid':
            start = size - 1
        else:
            start = 0

        if (start, strides) == (0, 1):
            return mask

        return mask[:, start::strides]

    def _compute_mask_with_threshold(self, mask):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        mask = tf.expand_dims(tf.cast(mask, self.dtype), axis=channel_axis)
        # shape (N, W, 1) or (N, 1, W)

        if self.padding == 'causal':
            mask = tf.pad(mask, self._compute_causal_padding())
        true_count = self._mask_op(mask, self.mask_kernel)  # shape (N, W, 1) or (N, 1, W)
        true_count = tf.squeeze(true_count, axis=channel_axis)  # shape (N, W)

        return tf.greater(true_count, self.mask_threshold - 0.1)  # avoid rounding error
