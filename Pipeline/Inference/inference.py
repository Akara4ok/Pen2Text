"""Abstract class for inference"""

import abc
from path import Path


class DataLoader(metaclass=abc.ABCMeta):
    """Abstract class for inference"""

    def __init__(self, preprocessor) -> None:
        self.preprocessor = preprocessor
        