from abc import ABC, abstractmethod
from typing import Any, List, Union

from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.core import Document


class DocumentMetric(ABC):
    """This defines the interface for a document metric."""

    def reset(self) -> None:
        """Any reset logic that needs to be performed before the metric is called again."""
        pass

    def __call__(
        self, document: Union[List[Document], Document, Dataset, IterableDataset]
    ) -> None:
        if isinstance(document, (list, Dataset, IterableDataset)):
            for doc in document:
                self(doc)
        elif isinstance(document, Document):
            self._update(document)
        else:
            raise Exception(f"document has unknown type: {type(document)}")

    def values(self, reset: bool = True) -> Any:
        metric_values = self._values()
        if reset:
            self.reset()
        return metric_values

    @abstractmethod
    def _update(self, document: Document) -> None:
        """This method is called to update the metric with the new document."""
        ...

    @abstractmethod
    def _values(self) -> Any:
        """This method is called to get the metric values."""
        ...
