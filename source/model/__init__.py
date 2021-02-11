__all__ = ['cnn']

from os import path
import sys

root = path.abspath('../..')

if root not in sys.path:
    sys.path.append(root)

from source.model.cnn import cnn_model, callbacks