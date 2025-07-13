from typing import Dict, Iterable, Optional, Sequence, Type, TypeVar

from pie_datasets import Dataset, DatasetDict, IterableDataset
from pytorch_ie.core import Document

from src.serializer.interface import DocumentSerializer
from src.utils.logging_utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


class JsonSerializer(DocumentSerializer):
    def __init__(self, **kwargs):
        self.default_kwargs = kwargs

    @classmethod
    def write(
        cls,
        documents: Iterable[Document],
        path: str,
        split: str = "train",
        append: bool = False,
    ) -> Dict[str, str]:

        if not isinstance(documents, (Dataset, IterableDataset)):
            if not isinstance(documents, Sequence):
                documents = IterableDataset.from_documents(documents)
            else:
                documents = Dataset.from_documents(documents)

        dataset_dict = DatasetDict({split: documents})
        dataset_dict.to_json(path=path, mode="a" if append else "w")
        return {"path": path, "split": split}

    @classmethod
    def read(
        cls,
        path: str,
        document_type: Optional[Type[D]] = None,
        split: Optional[str] = None,
    ) -> Dataset[Document]:
        dataset_dict = DatasetDict.from_json(
            data_dir=path, document_type=document_type, split=split
        )
        if split is not None:
            return dataset_dict[split]
        if len(dataset_dict) == 1:
            return dataset_dict[list(dataset_dict.keys())[0]]
        raise ValueError(f"multiple splits found in dataset_dict: {list(dataset_dict.keys())}")

    def read_with_defaults(self, **kwargs) -> Sequence[D]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.read(**all_kwargs)

    def write_with_defaults(self, **kwargs) -> Dict[str, str]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.write(**all_kwargs)

    def __call__(
        self, documents: Iterable[Document], append: bool = False, **kwargs
    ) -> Dict[str, str]:
        return self.write_with_defaults(documents=documents, append=append, **kwargs)
