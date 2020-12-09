from library.tf_keras_zoo.networks import Model
from tensorflow.python.keras.engine.base_layer import InputSpec

from .masking import MaskConv1D


class ResBlock(Model):

    def __init__(self, kernel_size=3, activation='elu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation

        self._input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    @property
    def input_spec(self):
        return self._input_spec

    def build(self, input_shape):
        filters = input_shape[-1].value
        self.conv1, self.conv2 = [
            MaskConv1D(
                filters=filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
                kernel_initializer='lecun_normal',
                padding='same',
            ) for _ in range(2)
        ]
        self._input_spec.axes = {2: filters}
        super().build(input_shape)

    def call(self, inputs, mask=None):
        x = self.conv1(inputs, mask=mask)
        x = self.conv2(x, mask=mask)
        return x + inputs

    def compute_mask(self, inputs, mask):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape
