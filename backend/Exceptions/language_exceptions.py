""" Classes for language exceptions """

import sys
sys.path.append('Exceptions')
from exceptions import Error

class NotSupportLanguageException(Error):
    """ Not supported language exceptions """

    def __init__(self, status: int = 400, message: str = "This language is currently not supported") -> None:
        super().__init__(status, message)
