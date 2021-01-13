import torch
from torch.nn import Embedding, Linear, Module

from .sequence_modeling import TokenSequence
from .interfaces import ModuleInterface


class Discriminator(Module, ModuleInterface):

    scope = 'Discriminator'

    def __init__(self, network: Module, embedder: Embedding):
        super().__init__()
        self.network = network
        self.embedder = embedder

        self.binary_output_layer = Linear(
            in_features=network(
                torch.zeros([1, 20, embedder.embedding_dim]),
            ).shape[-1],
            out_features=1,
        )

    def score_samples(self, samples: TokenSequence) -> torch.Tensor:
        word_vecs = self.get_embedding(samples.ids)
        return self.score_word_vector(word_vecs, samples.mask)

    def get_embedding(self, word_ids) -> torch.Tensor:
        return self.embedder(word_ids)

    def score_word_vector(self, word_vecs, mask=None) -> torch.Tensor:
        features = self.network(word_vecs, mask=mask)
        return self.binary_output_layer(features)

    @property
    def networks(self):
        return [self.embedder, self.network, self.binary_output_layer]

    @property
    def embedding_matrix(self):
        return self.embedder.weight
