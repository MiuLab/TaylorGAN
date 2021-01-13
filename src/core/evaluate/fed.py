from functools import lru_cache
from typing import List

import numpy as np

from library.utils import batch_generator


class FEDCalculator:

    def __init__(self, hub_url, references):
        self.fed_model = FEDModel.download_from(hub_url)
        self.encoded_references = self.fed_model.encode(references)

    def calculate_fed_score(self, candidates: List[str]):
        encoded_candidates = self.fed_model.encode(candidates)
        return self.fed_model.compute_fed_score(
            self.encoded_references,
            encoded_candidates,
        )


class FEDModel:

    @classmethod
    @lru_cache(None)
    def download_from(cls, hub_url):
        return cls(hub_url)

    def __init__(self, hub_url):
        self.build_graph(hub_url)

    def build_graph(self, hub_url):
        import tensorflow as tf
        import tensorflow_hub as hub

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Universal Sentence Encoder
            sentence_encoder = hub.Module(hub_url)
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            self.encoded_sentences = sentence_encoder(self.text_input)
            # FED
            self.fed_input_1 = tf.placeholder(tf.float32)
            self.fed_input_2 = tf.placeholder(tf.float32)
            self.fed_score = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(
                self.fed_input_1,
                self.fed_input_2,
            )
            init_op = tf.group(
                tf.variables_initializer(self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
                tf.tables_initializer(),
            )

        self.graph.finalize()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(init_op)

    def encode(self, sentences: List[str], batch_size: int = 64):
        encoded_sentences = [
            self.sess.run(
                self.encoded_sentences,
                feed_dict={self.text_input: batch_sentences},
            )
            for batch_sentences in batch_generator(sentences, batch_size)
        ]
        return np.concatenate(encoded_sentences, axis=0)

    def compute_fed_score(self, references, candidates):
        return self.sess.run(
            self.fed_score,
            feed_dict={
                self.fed_input_1: references,
                self.fed_input_2: candidates,
            },
        )
