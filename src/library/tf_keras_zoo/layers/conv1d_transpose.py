import operator

from tensorflow.contrib.nn import conv1d_transpose

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.utils import conv_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn


class Conv1DTranspose(Conv1D):

    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            padding='valid',
            output_padding=None,
            data_format=None,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
        ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs,
        )
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 1, 'output_padding',
            )
            if any(map(operator.le, self.strides, self.output_padding)):
                raise ValueError(
                    f'Stride {self.strides} must be greater than '
                    f'output padding {self.output_padding}',
                )

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 3:
            raise ValueError(f'Inputs should have rank 3. Received input shape: {input_shape}')
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError(
                'The channel dimension of the inputs should be defined. Found `None`.',
            )
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            w_axis = 2
        else:
            w_axis = 1

        width = inputs_shape[w_axis]
        kernel_w, = self.kernel_size
        stride_w, = self.strides

        if self.output_padding is None:
            out_pad_w = None
        else:
            out_pad_w, = self.output_padding

        # Infer the dynamic output shape:
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[0],
        )
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_width)
        else:
            output_shape = (batch_size, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = conv1d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            stride=stride_w,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, ndim=3),
        )

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, w_axis = 1, 2
        else:
            c_axis, w_axis = 2, 1

        kernel_w, = self.kernel_size
        stride_w, = self.strides

        if self.output_padding is None:
            out_pad_w = None
        else:
            out_pad_w, = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[0],
        )
        return tensor_shape.TensorShape(output_shape)
