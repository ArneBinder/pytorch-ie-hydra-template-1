from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple, Union

from pytorch_ie.core import Document, DocumentMetric
from transformers import AutoTokenizer


def _flatten_dict_gen(d, parent_key: Tuple[str, ...] = ()) -> Generator:
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            yield from dict(_flatten_dict_gen(v, new_key)).items()
        else:
            yield new_key, v


def flatten_dict(d: Dict[str, Any]) -> Dict[Tuple[str, ...], Any]:
    return dict(_flatten_dict_gen(d))


def unflatten_dict(d: Dict[Tuple[str, ...], Any]) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested keys.

    Example:
    ```python
    >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e"): 3}
    >>> unflatten_dict(d)
    {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
    ```
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if len(k) == 0:
            if len(result) > 1:
                raise ValueError("Cannot unflatten dictionary with multiple root keys.")
            return v
        current = result
        for key in k[:-1]:
            current = current.setdefault(key, {})
        current[k[-1]] = v
    return result


class DocumentStatistic(DocumentMetric):
    """A special type of metric that collects statistics from a document.

    Usage:

    ```python
    from pytorch_ie import DatasetDict

    dataset = DatasetDict.load_dataset("pie/conll2003")
    statistic = DocumentTokenCounter(tokenizer_name_or_path="bert-base-cased", field="text")
    values = statistic(dataset)
    ```
    """

    def reset(self) -> None:
        self._values: List[Any] = []

    @abstractmethod
    def _collect(self, doc: Document) -> Any:
        """Collect any values from a document."""

    def _update(self, document: Document) -> None:
        values = self._collect(document)
        self._values.append(values)

    def _compute(self) -> Any:
        """We just integrate the values by creating lists for each leaf of the (nested)
        dictionary."""
        stats = defaultdict(list)
        for metric_result in self._values:
            if isinstance(metric_result, dict):
                measure_result_flat = flatten_dict(metric_result)
                for k, v in measure_result_flat.items():
                    if isinstance(v, list):
                        stats[k].extend(v)
                    else:
                        stats[k].append(v)
            else:
                if isinstance(metric_result, list):
                    stats[()].extend(metric_result)
                else:
                    stats[()].append(metric_result)
        return unflatten_dict(dict(stats))


class DocumentTokenCounter(DocumentStatistic):
    """Counts the number of tokens in a field by tokenizing it with a Huggingface tokenizer.

    The field should be a string.
    """

    def __init__(self, tokenizer_name_or_path: str, field: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.kwargs = kwargs
        self.field = field

    def _collect(self, doc: Document) -> int:
        text = getattr(doc, self.field)
        encodings = self.tokenizer(text, **self.kwargs)
        tokens = encodings.tokens()
        return len(tokens)


class DocumentFieldLengthCounter(DocumentStatistic):
    """Counts the length of a field.

    The field should be a list of sized elements.
    """

    def __init__(self, field: str):
        self.field = field

    def _collect(self, doc: Document) -> int:
        field_obj = getattr(doc, self.field)
        return len(field_obj)


class DocumentSubFieldLengthCounter(DocumentStatistic):
    """Counts the length of a subfield in a field."""

    def __init__(self, field: str, subfield: str):
        self.field = field
        self.subfield = subfield

    def _collect(self, doc: Document) -> List[int]:
        field_obj = getattr(doc, self.field)
        lengths = []
        for entry in field_obj:
            subfield_obj = getattr(entry, self.subfield)
            lengths.append(len(subfield_obj))
        return lengths


class DocumentSpanLengthCounter(DocumentStatistic):
    """Counts the length of spans in a field.

    The field should be a list of elements with a label, a start and end attribute.
    """

    def __init__(self, field: str):
        self.field = field

    def _collect(self, doc: Document) -> Dict[str, List[int]]:
        field_obj = getattr(doc, self.field)
        counts = defaultdict(list)
        for elem in field_obj:
            counts[elem.label].append(elem.end - elem.start)
        return dict(counts)


class DummyCounter(DocumentStatistic):
    """A dummy counter that always returns 1.

    Can be used to count the number of documents.
    """

    def _collect(self, doc: Document) -> int:
        return 1


class LabelCounter(DocumentStatistic):
    """A counter that counts the number of labels in a field.

    The field should be a list of elements with a label attribute.
    """

    def __init__(self, field: str):
        self.field = field

    def _collect(self, doc: Document) -> Dict[str, int]:
        field_obj = getattr(doc, self.field)
        counts: Dict[str, int] = defaultdict(lambda: 1)
        for elem in field_obj:
            counts[elem.label] += 1
        return dict(counts)
