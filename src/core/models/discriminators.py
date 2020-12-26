import tensorflow as tf

from library.tf_keras_zoo.layers import Dense, Embedding, Layer

from .sequence_modeling import TokenSequence
from .interfaces import ModuleInterface


class Discriminator(ModuleInterface):

    scope = 'Discriminator'

    def __init__(self, network: Layer, embedder: Embedding, name: str = None):
        super().__init__(name)
        self.network = network
        self.embedder = embedder
        self.binary_output_layer = Dense(units=1)

    def score_samples(self, samples: TokenSequence) -> tf.Tensor:
        word_vecs = self.get_embedding(samples.ids)
        return self.score_word_vector(word_vecs, samples.mask)

    def get_embedding(self, word_ids) -> tf.Tensor:
        with tf.keras.backend.name_scope(self.scope):
            return self.embedder(word_ids)

    def score_word_vector(self, word_vecs, mask=None) -> tf.Tensor:
        with tf.keras.backend.name_scope(self.scope):
            features = self.network(word_vecs, mask=mask)
            logits = self.binary_output_layer(features)
        return logits

    @property
    def networks(self):
        return [self.embedder, self.network, self.binary_output_layer]

    @property
    def embedding_matrix(self):
        return self.embedder.total_embeddings
