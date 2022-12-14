""" Classes for exceptions """

class Error(Exception):
    """Base class for other exceptions"""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(self.message)