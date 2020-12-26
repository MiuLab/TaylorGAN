import tensorflow as tf


class LookAhead(tf.train.Optimizer):

    '''Reference: https://arxiv.org/abs/1907.08610'''

    def __init__(
            self,
            optimizer: tf.train.Optimizer,
            alpha: float = 0.5,
            explore_steps: int = 5,
            use_locking: bool = False,
            name: str = 'LookAhead',
        ):
        super().__init__(use_locking, name)
        self.optimizer = optimizer
        self.alpha = alpha
        self.explore_steps = explore_steps
        self.ema = tf.train.ExponentialMovingAverage(
            decay=1. - alpha,
            name="LookAheadSlowVariables",
        )

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()  # initial 0

        # global_step will be updated here
        update_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        var_list = [v for g, v in grads_and_vars if g is not None]

        with tf.control_dependencies([update_op]):
            finish_op = tf.cond(
                tf.equal(
                    tf.mod(global_step, self.explore_steps),
                    0,
                ),
                lambda: self._slow_fast_updates(var_list),
                tf.no_op,
                name=name,
            )

        return finish_op

    def _slow_fast_updates(self, var_list):
        with tf.control_dependencies([self.ema.apply(var_list)]):  # update slow
            return tf.group(*[
                var.assign(
                    self.ema.average(var),
                    use_locking=self._use_locking,
                )  # synchronize fast by slow
                for var in var_list
            ])
