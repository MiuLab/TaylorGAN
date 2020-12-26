from typing import Dict, List

import numpy as np
import umsgpack

from uttut.pipeline.pipe import Pipe
from uttut.pipeline.ops import Operator

from library.utils import JSONSerializableMixin, ObjectWrapper


class UttutPipeline(JSONSerializableMixin, ObjectWrapper):

    def __init__(self, ops: List[Operator] = ()):
        pipe = Pipe()
        for op in ops:
            pipe.add_op(op)
        self._pipe = pipe
        super().__init__(pipe)

    @classmethod
    def from_config(cls, config):
        pipe = cls()
        pipe._pipe = Pipe.deserialize(config)
        return pipe

    def get_config(self):
        return self._pipe.serialize()

    def transform_sequence(self, sequence):
        return self._pipe.transform_sequence(sequence)[0]

    def summary(self):
        self._pipe.summary()
        print()


class WordEmbeddingCollection:

    DTYPE = np.float32
    UNK = '<unk>'

    def __init__(self, token2index: Dict[str, int], vectors: List[List[float]]):
        self.token2index = token2index
        self.vectors = np.asarray(vectors, self.DTYPE)

    @classmethod
    def load_msg(cls, path: str):
        with open(path, "rb") as f_in:
            params = umsgpack.unpack(f_in)
        return cls(token2index=params['token2index'], vectors=params['vector'])

    def get_matrix_of_tokens(self, token_list: List[str]):
        return np.array(list(map(self.get_vector_of_token, token_list)))

    def get_vector_of_token(self, token: str):
        if token in self.token2index:
            return self.vectors[self.token2index[token]]
        elif self.UNK in self.token2index:
            return self.vectors[self.token2index[self.UNK]]
        else:
            return np.zeros_like(self.vectors[0])
