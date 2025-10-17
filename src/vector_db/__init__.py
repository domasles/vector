"""
Vector Database Library
VECTOR = "VECTOR Encodes Coordinates To Optimize Retrieval"

A coordinate-based n-dimensional database system with O(1) lookups.
"""

from .application import VectorDB
from .meta import __version__

__all__ = ['VectorDB', '__version__']
