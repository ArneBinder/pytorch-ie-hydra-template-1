from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from pytorch_ie.core import Document


class DocumentSerializer(ABC):
    """This defines the interface for a document serializer.

    The serializer should not return the serialized documents, but write them to disk and instead
    return some metadata such as the path to the serialized documents.
    """

    @abstractmethod
    def __init__(self, split: Optional[str] = None, **kwargs) -> None: ...

    @abstractmethod
    def __call__(self, documents: Iterable[Document], append: bool = False, **kwargs) -> Any: ...
