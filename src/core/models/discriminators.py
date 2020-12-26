import tensorflow as tf

from library.tf_keras_zoo.layers import Dense, Embedding, Layer
from library.utils import ObjectWrapper

from .sequence_modeling import TokenSequence
from .interfaces import ModuleInterface


class DiscriminateResult(ObjectWrapper):

    def __init__(self, samples: TokenSequence, word_vecs: tf.Tensor, score: tf.Tensor):
        super().__init__(samples)
        self.samples = samples
        self.word_vecs = word_vecs
        self.score = score


class Discriminator(ModuleInterface):

    scope = 'Discriminator'

    def __init__(self, network: Layer, embedder: Embedding, name: str = None):
        super().__init__(name)
        self.network = network
        self.embedder = embedder
        self.binary_output_layer = Dense(units=1)

    def score_samples(self, samples: TokenSequence) -> DiscriminateResult:
        with tf.keras.backend.name_scope(self.scope):
            word_vecs = self.embedder(samples.ids)

        score = self.score_word_vector(word_vecs, samples.mask)
        return DiscriminateResult(samples=samples, word_vecs=word_vecs, score=score)

    def score_word_vector(self, word_vecs, mask=None) -> tf.Tensor:
        with tf.keras.backend.name_scope(self.scope):
            features = self.network(word_vecs, mask=mask)
            logits = self.binary_output_layer(features)
        return logits

    @property
    def networks(self):
        return [self.embedder, self.network, self.binary_output_layer]

    @property
    def embeddings(self):
        return self.embedder.total_embeddings
