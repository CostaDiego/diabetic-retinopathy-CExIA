__all__ = []

from os import path
import sys

root = path.abspath('../..')

if root not in sys.path:
    sys.path.append(root)