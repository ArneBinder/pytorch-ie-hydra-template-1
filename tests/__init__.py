import json
import pathlib
from typing import Dict, Type, TypeVar

from pytorch_ie.core import Document

TESTS_ROOT = pathlib.Path(__file__).parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"

T_doc = TypeVar("T_doc", bound=Document)


def _load_doc_from_json(path: str, document_type: Type[T_doc]) -> T_doc:
    doc_json = json.load(open(path))
    doc = document_type.fromdict(doc_json)
    return doc


def _config_to_str(cfg: Dict[str, str]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result
