# import sugar
from tensorflow.python.keras.engine.base_layer import InputSpec, Layer
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Dense, Lambda
from tensorflow.python.keras.layers.recurrent import LSTM, StackedRNNCells

from .conv1d_transpose import Conv1DTranspose
from .embeddings import Embedding
from .recurrent import GRUCell
