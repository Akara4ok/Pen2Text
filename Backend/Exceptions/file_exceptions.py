""" Classes for file exceptions """

import sys
sys.path.append('Exceptions')
from exceptions import Error

class FilesExceptions(Error):
    """ Base exceptions for files """
    def __init__(self, filename: str, status: int, message: str) -> None:
        self.filename = filename
        super().__init__(status, message)
    
    def __str__(self) -> str:
        return f'{self.message}. Filename: {self.filename}.'


class NotSupportContentTypeException(FilesExceptions):
    """ Not supported content type exceptions """

    def __init__(self, filename, supported_types: list, status: int = 415, message: str = "The format of file type is currently not supported") -> None:
        self.supported_types = supported_types
        super().__init__(filename, status, message)
    
    def __str__(self) -> str:
        return f'{super().__str__()} Supported types: {self.supported_types}'

class BadFileContentException(FilesExceptions):
    """ Bad file content exceptions """

    def __init__(self, filename, status: int = 400, message: str = "The content of file is not correct") -> None:
        super().__init__(filename, status, message)