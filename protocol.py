from typing import Protocol
from abc import abstractmethod


class Image(Protocol):
    "a 2d image"
    
    @property
    @abstractmethod
    def width(self) -> int:
        "the width of the image"
        raise NotImplementedError
    
    @property
    @abstractmethod
    def height(self) -> int:
        "the height of the image"
        raise NotImplementedError
    
