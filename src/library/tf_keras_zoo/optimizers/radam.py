import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import training_ops


class RAdamOptimizer(tf.train.AdamOptimizer):

    '''Reference: https://arxiv.org/abs/1908.03265v1'''

    # Add-On: create steps variable.
    def _create_slots(self, var_list):
        super()._create_slots(var_list)
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=1., name="steps", colocate_with=first_var)

    # Add-On: create rectified lr
    # Reference: https://arxiv.org/abs/1908.03265v1
    def _prepare(self):
        super()._prepare()
        N_sma_max = 2. / (1. - self._beta2_t) - 1.
        x = tf.sqrt(N_sma_max / ((N_sma_max - 2.) * (N_sma_max - 4.)))
        _, beta2_power, steps = self._get_beta_accumulators()
        N_sma = N_sma_max - 2. * steps * beta2_power / (1. - beta2_power)
        self.rectifier = x * tf.sqrt((N_sma - 4.) * (N_sma - 2.) / N_sma)
        self.rectified_lr = self._lr_t * self.rectifier
        self.condition = tf.greater(N_sma, 4.)

    # Override: return steps variable as well.
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/adam.py#L111-L118
    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph),
                    self._get_non_slot_variable("steps", graph=graph))

    # Override: implement conditional update.
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/adam.py#L149-L161
    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power, _ = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)

        return tf.cond(
            self.condition,
            lambda: training_ops.apply_adam(
                var, m, v,
                beta1_power,
                math_ops.cast(beta2_power, var.dtype.base_dtype),
                math_ops.cast(self.rectified_lr, var.dtype.base_dtype),  # instead of _lr_t
                beta1_t,
                beta2_t,
                math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
                grad, use_locking=self._use_locking).op,
            lambda: self._apply_dense_without_v(
                var, m, v,
                beta1_power,
                beta1_t, beta2_t,
                grad),
        )

    def _apply_dense_without_v(self, var, m, v, beta1_power, beta1, beta2, grad):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        update_v = v.assign_sub((1 - beta2) * (v - tf.square(grad)), use_locking=self._use_locking)

        var_update = training_ops.apply_momentum(
            var,
            m,
            lr_t / (1 - beta1_power),  # to adapt adam formula
            grad * (1 - beta1),  # to adapt adam formula
            beta1,
            use_locking=self._use_locking,
        ).op
        return control_flow_ops.group(var_update, update_v)

    # Override: implement conditional update.
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/adam.py#L163-L175
    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power, _ = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)

        return tf.cond(
            self.condition,
            lambda: training_ops.resource_apply_adam(
                var.handle, m.handle, v.handle,
                beta1_power,
                math_ops.cast(beta2_power, grad.dtype.base_dtype),
                math_ops.cast(self.rectified_lr, grad.dtype.base_dtype),  # instead of _lr_t
                beta1_t,
                beta2_t,
                math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
                grad, use_locking=self._use_locking),
            lambda: self._resource_apply_dense_without_v(
                var.handle, m.handle, v,
                beta1_power,
                beta1_t, beta2_t,
                grad),
        )

    def _resource_apply_dense_without_v(self, var, m, v, beta1_power, beta1, beta2, grad):
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        update_v = v.assign_sub((1 - beta2) * (v - tf.square(grad)), use_locking=self._use_locking)

        var_update = training_ops.resource_apply_momentum(
            var,
            m,
            lr_t / (1 - beta1_power),  # to adapt adam formula
            grad * (1 - beta1),  # to adapt adam formula
            beta1,
            use_locking=self._use_locking,
        )
        return control_flow_ops.group(var_update, update_v)

    # Override: implement conditional update.
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/adam.py#L177-L203
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power, _ = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = lr_t / (1 - beta1_power)

        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_sqrt = tf.sqrt(v_t)

        var_update = tf.cond(
            self.condition,
            lambda: var.assign_sub(
                (lr * tf.sqrt(1 - beta2_power) * self.rectifier) * m_t / (v_sqrt + epsilon_t),
                use_locking=self._use_locking),
            lambda: var.assign_sub(
                lr * m_t,
                use_locking=self._use_locking),
        )
        return control_flow_ops.group(var_update, m_t, v_t)

    # Override: update step as well.
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/adam.py#L221-L231
    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            b1_p, b2_p, step = self._get_beta_accumulators()
            with ops.colocate_with(b1_p):
                update_b1 = b1_p.assign(b1_p * self._beta1_t, use_locking=self._use_locking)
                update_b2 = b2_p.assign(b2_p * self._beta2_t, use_locking=self._use_locking)
                update_step = step.assign_add(1., use_locking=self._use_locking)

        return control_flow_ops.group(
            *update_ops, update_b1, update_b2, update_step,
            name=name_scope)
