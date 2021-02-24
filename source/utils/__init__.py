__all__ = ['structure']

from os import path
import sys

root = path.abspath('../..')

if root not in sys.path:
    sys.path.append(root)

from source.utils.structure import exists