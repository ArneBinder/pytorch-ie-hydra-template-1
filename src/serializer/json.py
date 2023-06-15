import json
import os
from typing import Dict, List, Sequence, Type, TypeVar

from pytorch_ie.core import Document

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
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def __call__(self, documents: Sequence[Document]) -> Dict[str, str]:
        return self.dump(documents=documents, path=self.path, **self.kwargs)

    @classmethod
    def dump(cls, documents: Sequence[Document], path: str, **kwargs) -> Dict[str, str]:
        realpath = os.path.realpath(path)
        log.info(f'serialize documents to "{realpath}" ...')
        dir_path = os.path.dirname(realpath)
        os.makedirs(dir_path, exist_ok=True)
        if as_json_lines(str(path)):
            with open(realpath, "w") as f:
                for doc in documents:
                    f.write(json.dumps(doc.asdict(), **kwargs) + "\n")
        else:
            with open(realpath, "w") as f:
                json.dump([doc.asdict() for doc in documents], fp=f, **kwargs)
        return {"path": realpath}

    @classmethod
    def read(cls, file_name: str, document_type: Type[D]) -> List[D]:
        documents = []
        if as_json_lines(str(file_name)):
            with open(file_name) as f:
                for line in f:
                    json_dict = json.loads(line)
                    documents.append(document_type.fromdict(json_dict))
        else:
            with open(file_name) as f:
                json_list = json.load(f)
            for json_dict in json_list:
                documents.append(document_type.fromdict(json_dict))
        return documents
