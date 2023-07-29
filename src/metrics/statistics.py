from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.core import Document
from pytorch_ie.data.common import EnterDatasetDictMixin, EnterDatasetMixin
from transformers import AutoTokenizer

from src.metrics.interface import DocumentMetric


def _flatten_dict_gen(d, parent_key: Tuple[str, ...] = ()) -> Generator:
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            yield from dict(_flatten_dict_gen(v, new_key)).items()
        else:
            yield new_key, v


class DocumentStatistic(DocumentMetric, EnterDatasetDictMixin, EnterDatasetMixin):
    """A special type of metric that collects statistics from a document. It should be used in
    combination with DatasetDict.map().

    Usage:

    ```python
    from pytorch_ie import DatasetDict

    dataset = DatasetDict.load_dataset("pie/conll2003")
    statistic = DocumentTokenCounter(tokenizer_name_or_path="bert-base-cased", field="text")
    # Note: it is important to disable caching (load_from_cache_file=False), otherwise the metric might
    # not be computed
    dataset.map(statistic, result_document_type=dataset.document_type, load_from_cache_file=False)
    values = statistic.compute()
    ```
    """

    def enter_dataset_dict(self, dataset_dict) -> None:
        self.reset()

    def reset(self) -> None:
        self._values: Dict[Optional[str], List[Any]] = defaultdict(list)

    def enter_dataset(
        self, dataset: Union[Dataset, IterableDataset], name: Optional[str] = None
    ) -> None:
        self.split_name = name

    def collect(self, doc: Document) -> Any:
        """Collects any values from a document."""

    def _update(self, document: Document) -> None:
        values = self.collect(document)
        self._values[self.split_name].append(values)

    def _compute(self) -> Dict[Tuple[str, ...], List[Any]]:
        stats = defaultdict(list)
        for split_name, split_metric_values in self._values.items():
            for metric_result in split_metric_values:
                if isinstance(metric_result, dict):
                    measure_result_flat = dict(_flatten_dict_gen(metric_result))
                    for k, v in measure_result_flat.items():
                        if isinstance(v, list):
                            stats[(split_name,) + k].extend(v)
                        else:
                            stats[(split_name,) + k].append(v)
                else:
                    if isinstance(metric_result, list):
                        stats[(split_name,)].extend(metric_result)
                    else:
                        stats[(split_name,)].append(metric_result)
        return dict(stats)


class DocumentTokenCounter(DocumentStatistic):
    """Counts the number of tokens in a field by tokenizing it with a Huggingface tokenizer.

    The field should be a string.
    """

    def __init__(self, tokenizer_name_or_path: str, field: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.kwargs = kwargs
        self.field = field

    def collect(self, doc: Document) -> int:
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

    def collect(self, doc: Document) -> int:
        field_obj = getattr(doc, self.field)
        return len(field_obj)


class DocumentSubFieldLengthCounter(DocumentStatistic):
    """Counts the length of a subfield in a field."""

    def __init__(self, field: str, subfield: str):
        self.field = field
        self.subfield = subfield

    def collect(self, doc: Document) -> List[int]:
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

    def collect(self, doc: Document) -> Dict[str, List[int]]:
        field_obj = getattr(doc, self.field)
        counts = defaultdict(list)
        for elem in field_obj:
            counts[elem.label].append(elem.end - elem.start)
        return dict(counts)


class DummyCounter(DocumentStatistic):
    """A dummy counter that always returns 1.

    Can be used to count the number of documents.
    """

    def collect(self, doc: Document) -> int:
        return 1


class LabelCounter(DocumentStatistic):
    """A counter that counts the number of labels in a field.

    The field should be a list of elements with a label attribute.
    """

    def __init__(self, field: str):
        self.field = field

    def collect(self, doc: Document) -> Dict[str, int]:
        field_obj = getattr(doc, self.field)
        counts: Dict[str, int] = defaultdict(lambda: 1)
        for elem in field_obj:
            counts[elem.label] += 1
        return dict(counts)
