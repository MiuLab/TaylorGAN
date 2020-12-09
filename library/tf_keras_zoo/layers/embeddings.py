import numpy as np
import tensorflow as tf
from typing import Sequence, Union

from tensorflow.python.keras.utils import tf_utils


class Embedding(tf.keras.layers.Embedding):

    def __init__(
            self,
            vocab_size,
            embeddings_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_index: Union[int, Sequence[int]] = None,
            input_length: int = None,
            dropout: float = None,
            **kwargs,
        ):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', tf.float32)
        super(tf.keras.layers.Embedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = self.vocab_size = vocab_size
        self.output_dim = self.embeddings_dim = embeddings_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

        self.mask_index = self._standardize_mask_index(mask_index)
        self.supports_masking = (mask_index is not None)
        self.input_length = input_length

        if not (dropout is None or 0. < dropout < 1.):
            raise ValueError(f"`dropout` should be in (0., 1.)! Found {dropout}")
        self.dropout = dropout

        self.auxiliary_tokens = 0
        self.extend_dims = 0
        self._constant = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self._constant:
            self.embeddings = tf.constant(self.embeddings_initializer.value)
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                trainable=self.trainable,
            )

        self.total_embeddings = self.embeddings

        if self.extend_dims > 0:
            self.extend_embeddings = self._force_trainable_add_weight(
                shape=(self.input_dim, self.extend_dims),
                name='extend_embeddings',
            )
            self.total_embeddings = tf.concat(
                [self.total_embeddings, self.extend_embeddings],
                axis=1,
                name='embeddings_with_extended_dims',
            )

        if self.auxiliary_tokens > 0:
            embeddings_dim = self.total_embeddings.shape[1].value
            self.auxiliary_embeddings = self._force_trainable_add_weight(
                shape=(self.auxiliary_tokens, embeddings_dim),
                name='auxiliary_embeddings',
            )
            self.total_embeddings = tf.concat(
                [self.total_embeddings, self.auxiliary_embeddings],
                axis=0,
                name='embeddings_with_auxiliary_tokens',
            )

        self.total_embeddings = tf.identity(self.total_embeddings, name='total_embeddings')
        self.built = True

    def _force_trainable_add_weight(self, **kwargs):
        # HACK, since Layer.add_weight will take
        # the intersection of trainable (in arg) and self.trainable
        # manually set self.trainable = True
        # to make sure weight is tracked by backend.
        original_trainable = self.trainable
        self.trainable = True
        weight = self.add_weight(**kwargs, trainable=True)
        self.trainable = original_trainable
        return weight

    @property
    def initializer(self):  # just for alias
        return self.embeddings_initializer

    @property
    def total_dim(self):
        return self.embeddings_dim + self.extend_dims

    @property
    def trainable_weights(self):
        # HACK in keras implementation, they consider layer.trainable as well,
        # it's ignored in this part.
        return self._trainable_weights

    @property
    def non_trainable_weights(self):
        # HACK in keras implementation, they consider layer.trainable as well,
        # it's ignored in this part.
        return self._non_trainable_weights

    @classmethod
    def from_weights(
            cls,
            weights: np.ndarray,
            mask_index: Union[int, Sequence[int]] = None,
            constant: bool = False,
            auxiliary_tokens: int = 0,
            extend_dims: int = 0,
            dropout: float = None,
            **kwargs,
        ):
        '''Create a Embedding Layer by pre-defined matrix of shape (vocab_size, dimension).

        Args:
            weights: numpy array of shape (vocab_size, dimension).
            mask_index: Which input index would be masked out in output._keras_mask.
            constant: If True, embeddings would be created as tf.constant instead of tf.Variable.
            auxiliary_tokens: Number of auxiliary trainable tokens,
                If > 0, tokens would be added `right after` the weights matrix.
                Useful when model needs some special tokens but they aren't pre-trained.
                Make sure these tokens' indices are in [vocab_size, vocab_size + auxiliary_tokens).

        Raises:
            ValueError: If weights is not a rank 2 matrix.

        Returns:
            layer: Instance of Embedding Layer.
        '''

        if weights.ndim != 2:
            raise ValueError(f"`weights` should be a rank 2 array! Recieved shape: {weights.shape}")
        dtype = weights.dtype
        if dtype not in (np.float32, np.float64):
            raise ValueError('`weights.dtype` should be float!')
        vocab_size, embeddings_dim = weights.shape
        initializer = tf.constant_initializer(weights)
        layer = cls(
            vocab_size=vocab_size,
            embeddings_dim=embeddings_dim,
            embeddings_initializer=initializer,
            mask_index=mask_index,
            dtype=dtype,
            dropout=dropout,
            **kwargs,
        )
        if constant:
            layer.trainable = False
            layer._constant = True
        layer.auxiliary_tokens = auxiliary_tokens
        layer.extend_dims = extend_dims
        return layer

    def _standardize_mask_index(self, mask_index):
        if mask_index is None:
            return None
        if isinstance(mask_index, Sequence):
            for idx in mask_index:
                self._valid_mask_index_int(idx)
            return set(mask_index)

        self._valid_mask_index_int(mask_index)
        return mask_index

    def _valid_mask_index_int(self, mask_index):
        if not isinstance(mask_index, int):
            raise ValueError("`mask_index` should be integer!")
        if not (0 <= mask_index < self.input_dim):
            raise ValueError("`mask_index` should be in range [0, input_dim)!")

    def call(self, inputs, mask=None, training=None):
        if inputs.dtype not in (tf.int32, tf.int64):
            inputs = tf.cast(inputs, tf.int32)

        if training is None:
            training = tf.keras.backend.learning_phase()

        if self.dropout is not None:
            # NOTE randomly drop token: row of embedding matrix
            # to avoid scaling by 1 / keep_prob, slightly modify `tf.nn.dropout`
            def dropped_embeddings():
                random_tensor = tf.random_uniform(
                    shape=(self.total_embeddings.shape[0].value, 1),
                    minval=1. - self.dropout,
                    maxval=2. - self.dropout,
                    dtype=self.total_embeddings.dtype,
                )
                # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                binary_tensor = tf.math.floor(random_tensor)
                return self.total_embeddings * binary_tensor

            embeddings = tf_utils.smart_cond(
                training,
                dropped_embeddings,
                lambda: tf.identity(self.total_embeddings),
            )
        else:
            embeddings = self.total_embeddings

        return tf.nn.embedding_lookup(embeddings, inputs)

    def compute_mask(self, inputs, mask):
        if self.mask_index is None:
            return mask

        if isinstance(self.mask_index, int):
            new_mask = tf.not_equal(inputs, self.mask_index)
        else:
            new_mask = tf.reduce_all(
                [tf.not_equal(inputs, idx) for idx in self.mask_index],
                axis=0,
            )

        if mask is None:
            return new_mask
        return tf.logical_and(mask, new_mask)

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embeddings_dim': self.embeddings_dim,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
            'mask_index': self.mask_index,
            'input_length': self.input_length,
            'auxiliary_tokens': self.auxiliary_tokens,
            'extend_dims': self.extend_dims,
            'dropout': self.dropout,
        }

    def copy(self):
        layer = self.__class__(
            vocab_size=self.vocab_size,
            embeddings_dim=self.embeddings_dim,
            embeddings_initializer=self.embeddings_initializer,
            trainable=self.trainable,
        )
        layer._constant = self._constant
        layer.extend_dims = self.extend_dims
        return layer


class OutputEmbedding(tf.keras.layers.Layer):

    def __init__(self, tie_embedder, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.tie_embedder = tie_embedder
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: tie_embedder.total_dim})

    @property
    def units(self):
        return self.tie_embedder.vocab_size

    def build(self, input_shape):
        if not self.tie_embedder.built:
            raise RuntimeError

        self.kernel = tf.matrix_transpose(self.tie_embedder.total_embeddings)  # shape (E, V)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units],
                initializer=tf.zeros_initializer(),
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        if inputs.shape.ndims > 2:
            logits = tf.tensordot(inputs, self.kernel, axes=[-1, 0])
        else:
            logits = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            logits = tf.nn.bias_add(logits, self.bias)
        return logits


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, dim, min_period=2., max_period=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def call(self, inputs):
        pos_emb = self._get_position_embedding_tensor(inputs.shape[1].value)
        batch_pos_emb = tf.tile(pos_emb[tf.newaxis], multiples=[tf.shape(inputs)[0], 1, 1])
        outputs = tf.concat([inputs, batch_pos_emb], axis=2)
        return outputs

    def _get_position_embedding_tensor(self, maxlen):
        max_period = 4. * maxlen if self.max_period is None else self.max_period
        wave_length = np.exp(np.linspace(
            np.log(self.min_period),
            np.log(max_period),
            num=self.dim,
        ))  # shape (D)
        pos_range = np.arange(maxlen)  # shape (T)
        theta = 2 * np.pi * pos_range[:, np.newaxis] / wave_length  # shape (T, D)
        return tf.constant(np.sin(theta), dtype=self.dtype)
