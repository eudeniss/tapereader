"""
Módulo de console visual do TapeReader
"""

from .display import TapeReaderConsole
from .formatter import SignalFormatter, TableFormatter
from .templates import ConsoleTemplates

__all__ = [
    'TapeReaderConsole',
    'SignalFormatter',
    'TableFormatter',
    'ConsoleTemplates'
]