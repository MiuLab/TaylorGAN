import tensorflow as tf

from core.models import ModuleInterface
from .base import Regularizer, LossCollection


class VariableRegularizer(Regularizer):

    def __call__(self, generator=None, discriminator=None, **kwargs) -> LossCollection:
        if generator and discriminator:
            raise TypeError
        loss = self.compute_loss(module=generator or discriminator)
        return LossCollection(self.coeff * loss, **{self.loss_name: loss})


class EmbeddingRegularizer(VariableRegularizer):

    loss_name = 'embedding'

    def __init__(self, coeff: float, max_norm: float = 0.):
        super().__init__(coeff)
        self.max_norm = max_norm

    def compute_loss(self, module: ModuleInterface):
        embedding_L2_loss = tf.reduce_sum(tf.square(module.embedding_matrix), axis=1)  # shape (V, )
        if self.max_norm:
            embedding_L2_loss = tf.maximum(embedding_L2_loss - self.max_norm ** 2, 0)
        return tf.reduce_mean(embedding_L2_loss) / 2  # shape ()


class SpectralRegularizer(VariableRegularizer):

    loss_name = 'spectral'

    def compute_loss(self, module: ModuleInterface):
        spectral_L2_list, update_list = [], []
        for kernel in filter(
            lambda v: 'kernel' in v.name and v.shape.ndims >= 2,
            module.trainable_variables,
        ):
            sn, update = self._get_spectral_norm(kernel)
            spectral_L2_list.append(tf.square(sn))
            update_list.append(update)

        with tf.control_dependencies(update_list):
            spectral_L2_sum = tf.add_n(spectral_L2_list)

        return spectral_L2_sum / 2

    def _get_spectral_norm(self, kernel: tf.Variable):
        if kernel.shape.ndims > 2:
            kernel_matrix = tf.reshape(kernel, [-1, kernel.shape[-1].value])
        else:
            kernel_matrix = kernel  # shape (U, V)
        u = tf.get_variable(
            name=f'{kernel.op.name}/left_singular_vector',
            shape=(kernel_matrix.shape[0].value, ),
            initializer=tf.keras.initializers.lecun_normal(),  # unit vector
            trainable=False,
            dtype=kernel_matrix.dtype,
        )  # shape (U)
        v = tf.stop_gradient(
            tf.nn.l2_normalize(tf.linalg.matvec(kernel_matrix, u, transpose_a=True)),
        )  # shape (V)
        Wv = tf.linalg.matvec(kernel_matrix, v)  # shape (U)
        new_u = tf.stop_gradient(tf.nn.l2_normalize(Wv))  # shape (U)

        spectral_norm = tf.tensordot(new_u, Wv, axes=1)
        update_u = tf.assign(u, new_u)
        return spectral_norm, update_u
