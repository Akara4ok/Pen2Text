""" Classes for pipeline exceptions """

import sys
sys.path.append('Exceptions')
from exceptions import Error

class PipelineException(Error):
    """ Pipeline error """

    def __init__(self, index: int, filename:str = "", status: int = 500, message: str = "The model pipeline has been crashed") -> None:
        self.filename = filename
        self.index = index
        super().__init__(status, message)

    def __str__(self) -> str:
        return f'{self.message}. Filename: {self.filename}.'