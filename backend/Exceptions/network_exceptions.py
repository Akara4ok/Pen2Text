""" Classes for network exceptions """

import sys
sys.path.append('Exceptions')
from exceptions import Error

class NotSupportNetworkException(Error):
    """ Not supported network exceptions """

    def __init__(self, status: int = 400, message: str = "This network is currently not supported") -> None:
        super().__init__(status, message)
