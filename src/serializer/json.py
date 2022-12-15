import json
import os
from typing import Sequence

from pytorch_ie.core import Document

from src import utils

log = utils.get_pylogger(__name__)


class JsonSerializer:
    def __init__(self, path: str, as_lines: bool = True, **kwargs):
        self.path = path
        self.as_lines = as_lines
        self.kwargs = kwargs

    def __call__(self, documents: Sequence[Document]):

        realpath = os.path.realpath(self.path)
        log.info(f'serialize documents to "{realpath}" ...')
        dir_path = os.path.dirname(realpath)
        os.makedirs(dir_path, exist_ok=True)

        if self.as_lines:
            with open(self.path, "w") as f:
                for doc in documents:
                    f.write(json.dumps(doc.asdict(), **self.kwargs) + "\n")
        else:
            with open(self.path, "w") as f:
                json.dump([doc.asdict() for doc in documents], fp=f, **self.kwargs)
