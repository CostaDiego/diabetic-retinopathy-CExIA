__all__ = ['preProcess', 'model']

from os import path
import sys

root = path.abspath('..')

if root not in sys.path:
    sys.path.append(root)

from source import model
from source import preProcess