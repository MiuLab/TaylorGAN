import abc

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine.training import Model as keras_Model
from tensorflow.python.keras.utils import generic_utils


class Model(keras_Model, abc.ABC):

    # HACK override: remove pre-call part!!
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/engine/network.py#L690-L784
    # just copy/paste the source code and remove L732-L781

    @base_layer.default
    def build(self, input_shape):
        # source code L752
        if self._is_graph_network:
            self.built = True
            return

        # If subclass network
        if input_shape is None:
            raise ValueError(
                'Input shape must be defined when calling build on a '
                'model subclass network.',
            )
        valid_types = (tuple, list, tensor_shape.TensorShape)
        if not isinstance(input_shape, valid_types):
            raise ValueError(
                'Specified input shape is not one of the valid types. '
                'Please specify a batch input shape of type tuple or '
                'list of input shapes. User provided '
                f'input type: {type(input_shape)}',
            )

        # remove L732-L781

        # source code L782
        if self._layers:
            self._track_layers(self._layers)
        self.built = True

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None):
        pass

    # HACK override: fix output._keras_mask setting and remove the try/except part.
    # don't use sublayer outputs keras_mask if implement compute_mask method.
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/engine/base_layer.py#L1350-L1371
    def _set_mask_metadata(self, inputs, outputs, previous_mask):
        output_list = generic_utils.to_list(outputs)
        # call compute_mask, even if all _keras_mask already computed!!!!!
        if hasattr(self, 'compute_mask'):
            output_mask = self.compute_mask(inputs, previous_mask)
        else:
            # Fix source code L1358
            output_mask = [getattr(x, '_keras_mask', None) for x in output_list]

        if output_mask is None:
            output_mask_list = [None] * len(output_list)
        else:
            output_mask_list = generic_utils.to_list(output_mask)

        for x, m in zip(output_list, output_mask_list):
            # Fix source code 1368, don't except Attribute Error
            x._keras_mask = m
