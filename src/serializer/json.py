import json
import os
from typing import Dict, List, Optional, Sequence, Type, TypeVar

from pie_datasets.core.dataset_dict import METADATA_FILE_NAME
from pytorch_ie.core import Document
from pytorch_ie.utils.hydra import resolve_optional_document_type, serialize_document_type

from src.serializer.interface import DocumentSerializer
from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


def as_json_lines(file_name: str) -> bool:
    if file_name.lower().endswith(".jsonl"):
        return True
    elif file_name.lower().endswith(".json"):
        return False
    else:
        raise Exception(f"unknown file extension: {file_name}")


class JsonSerializer(DocumentSerializer):
    def __init__(self, **kwargs):
        self.default_kwargs = kwargs

    @classmethod
    def write(
        cls,
        documents: Sequence[Document],
        path: str,
        file_name: str = "documents.jsonl",
        metadata_file_name: str = METADATA_FILE_NAME,
        split: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        realpath = os.path.realpath(path)
        log.info(f'serialize documents to "{realpath}" ...')
        os.makedirs(realpath, exist_ok=True)

        # dump metadata including the document_type
        if len(documents) == 0:
            raise Exception("cannot serialize empty list of documents")
        document_type = type(documents[0])
        metadata = {"document_type": serialize_document_type(document_type)}
        full_metadata_file_name = os.path.join(realpath, metadata_file_name)
        if os.path.exists(full_metadata_file_name):
            log.warning(
                f"metadata file {full_metadata_file_name} already exists, "
                "it will be overwritten!"
            )
        with open(full_metadata_file_name, "w") as f:
            json.dump(metadata, f, indent=2)

        if split is not None:
            realpath = os.path.join(realpath, split)
            os.makedirs(realpath, exist_ok=True)
        full_file_name = os.path.join(realpath, file_name)
        if as_json_lines(file_name):
            with open(full_file_name, "w") as f:
                for doc in documents:
                    f.write(json.dumps(doc.asdict(), **kwargs) + "\n")
        else:
            with open(full_file_name, "w") as f:
                json.dump([doc.asdict() for doc in documents], fp=f, **kwargs)
        return {"path": realpath, "file_name": file_name, "metadata_file_name": metadata_file_name}

    @classmethod
    def read(
        cls,
        path: str,
        document_type: Optional[Type[D]] = None,
        file_name: str = "documents.jsonl",
        metadata_file_name: str = METADATA_FILE_NAME,
        split: Optional[str] = None,
    ) -> List[D]:
        realpath = os.path.realpath(path)
        log.info(f'load documents from "{realpath}" ...')

        # try to load metadata including the document_type
        full_metadata_file_name = os.path.join(realpath, metadata_file_name)
        if os.path.exists(full_metadata_file_name):
            with open(full_metadata_file_name) as f:
                metadata = json.load(f)
            document_type = resolve_optional_document_type(metadata.get("document_type"))

        if document_type is None:
            raise Exception("document_type is required to load serialized documents")

        if split is not None:
            realpath = os.path.join(realpath, split)
        full_file_name = os.path.join(realpath, file_name)
        documents = []
        if as_json_lines(str(file_name)):
            with open(full_file_name) as f:
                for line in f:
                    json_dict = json.loads(line)
                    documents.append(document_type.fromdict(json_dict))
        else:
            with open(full_file_name) as f:
                json_list = json.load(f)
            for json_dict in json_list:
                documents.append(document_type.fromdict(json_dict))
        return documents

    def read_with_defaults(self, **kwargs) -> List[D]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.read(**all_kwargs)

    def write_with_defaults(self, **kwargs) -> Dict[str, str]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.write(**all_kwargs)

    def __call__(self, documents: Sequence[Document], **kwargs) -> Dict[str, str]:
        return self.write_with_defaults(documents=documents, **kwargs)
