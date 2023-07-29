from collections import defaultdict
from typing import Dict, List

from pytorch_ie.core import Document
from transformers import AutoTokenizer


class DocumentTokenCounter:
    def __init__(self, tokenizer_name_or_path: str, field: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.kwargs = kwargs
        self.field = field

    def __call__(self, doc: Document) -> int:
        text = getattr(doc, self.field)
        encodings = self.tokenizer(text, **self.kwargs)
        tokens = encodings.tokens()
        return len(tokens)


class DocumentFieldLengthCounter:
    def __init__(self, field: str):
        self.field = field

    def __call__(self, doc: Document) -> int:
        field_obj = getattr(doc, self.field)
        return len(field_obj)


class DocumentSubFieldLengthCounter:
    def __init__(self, field: str, subfield: str):
        self.field = field
        self.subfield = subfield

    def __call__(self, doc: Document) -> List[int]:
        field_obj = getattr(doc, self.field)
        lengths = []
        for entry in field_obj:
            subfield_obj = getattr(entry, self.subfield)
            lengths.append(len(subfield_obj))
        return lengths


class DocumentSpanLengthCounter:
    def __init__(self, field: str):
        self.field = field

    def __call__(self, doc: Document) -> Dict[str, List[int]]:
        field_obj = getattr(doc, self.field)
        counts = defaultdict(list)
        for elem in field_obj:
            counts[elem.label].append(elem.end - elem.start)
        return dict(counts)


class DummyCounter:
    def __call__(self, doc: Document) -> int:
        return 1


class LabelCounter:
    def __init__(self, field: str):
        self.field = field

    def __call__(self, doc: Document) -> Dict[str, int]:
        field_obj = getattr(doc, self.field)
        counts: Dict[str, int] = defaultdict(lambda: 1)
        for elem in field_obj:
            counts[elem.label] += 1
        return dict(counts)
