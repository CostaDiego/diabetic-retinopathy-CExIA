__all__ = ['cnn', 'callback']

from os import path
import sys

root = path.abspath('../..')

if root not in sys.path:
    sys.path.append(root)

from source.model.cnn import cnn_model
from source.model.callback import earlyStopping, tensorBoard, callbacks