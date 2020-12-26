import numpy as np
import tensorflow as tf

from tqdm import tqdm

from library.utils import batch_generator


class LSTMLMCalculator:

    def __init__(
            self,
            save_path: str,
            batch_size: int = 64,
            load_weight: bool = True,
        ):
        self.batch_size = batch_size
        self.save_path = save_path
        self.build_model(save_path, load_weight)

    def build_model(self, save_path, load_weight):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph, config=config)
        with graph.as_default(), self.sess.as_default():
            self.saver = tf.train.import_meta_graph(f'{save_path}.meta')
            self.input_data = graph.get_tensor_by_name("input/data:0")
            self.loss = graph.get_tensor_by_name("output/loss:0")
            self.perplexity_tensor = graph.get_tensor_by_name("output/perplexity:0")
            self.train_op = tf.train.AdamOptimizer(1e-4, name='Adam2').minimize(self.loss)
            if load_weight:
                self.saver.restore(self.sess, save_path)
            else:
                self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, save_path)

    def copy(self, load_weight: bool):
        return LSTMLMCalculator(
            save_path=self.save_path,
            batch_size=self.batch_size,
            load_weight=load_weight,
        )

    def fit(self, x: np.ndarray, epochs: int):
        for epoch in range(1, epochs + 1):
            with tqdm(
                batch_generator(x, batch_size=self.batch_size),
                desc=f'Epoch {epoch}',
                total=len(x) // self.batch_size,
                leave=False,
            ) as pbar:
                for batch_x in pbar:
                    _, loss = self.sess.run(
                        [self.train_op, self.loss],
                        feed_dict={self.input_data: batch_x},
                    )
                    pbar.set_postfix(loss=loss)

    def nll_loss(self, x: np.ndarray):
        return np.mean([
            self.sess.run(self.loss, feed_dict={self.input_data: batch_x})
            for batch_x in tqdm(
                batch_generator(x, self.batch_size, shuffle=False),
                total=len(x) // self.batch_size,
                leave=False,
            )
        ])
