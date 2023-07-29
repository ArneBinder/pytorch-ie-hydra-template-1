import dataclasses
from collections import defaultdict
from functools import partial
from typing import Dict

import pytest
from pytorch_ie import DatasetDict
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextBasedDocument

# from src.document.statistics import (
#    DocumentFieldLengthCounter,
#    DocumentTokenCounter,
#    DummyCounter,
#    LabelCounter,
# )
from src.metrics.statistics import (
    DocumentFieldLengthCounter,
    DocumentTokenCounter,
    DummyCounter,
    LabelCounter,
)
from src.utils.statistic_utils import prepare_data
from tests import FIXTURES_ROOT


@pytest.fixture
def dataset():
    @dataclasses.dataclass
    class Conll2003Document(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    return DatasetDict.from_json(
        data_dir=FIXTURES_ROOT / "dataset_dict" / "conll2003_extract",
        document_type=Conll2003Document,
    )


def test_prepare_data(dataset):
    prepared_data = prepare_data(dataset=dataset, metric=DummyCounter())
    assert prepared_data == {
        ("train",): [1, 1, 1],
        ("test",): [1, 1, 1],
        ("validation",): [1, 1, 1],
    }
    prepared_data = prepare_data(dataset=dataset, metric=LabelCounter(field="entities"))
    assert prepared_data == {
        ("train", "ORG"): [2],
        ("train", "MISC"): [3],
        ("train", "PER"): [2],
        ("train", "LOC"): [2],
        ("test", "LOC"): [2, 3],
        ("test", "PER"): [2, 2],
        ("validation", "ORG"): [2, 3],
        ("validation", "LOC"): [2],
        ("validation", "MISC"): [2],
        ("validation", "PER"): [2],
    }
    prepared_data = prepare_data(dataset=dataset, metric=DocumentFieldLengthCounter(field="text"))
    assert prepared_data == {
        ("train",): [48, 15, 19],
        ("test",): [57, 11, 40],
        ("validation",): [65, 17, 187],
    }


def label_counter(doc: Document, field: str) -> Dict[str, int]:
    field_obj = getattr(doc, field)
    counts: Dict[str, int] = defaultdict(lambda: 1)
    for elem in field_obj:
        counts[elem.label] += 1
    return dict(counts)


def document_field_length_collector(doc: Document, field: str) -> int:
    field_obj = getattr(doc, field)
    return len(field_obj)


def test_prepare_data_simple(dataset):
    prepared_data = prepare_data(dataset=dataset, metric=lambda doc: 1)
    assert prepared_data == {
        ("train",): [1, 1, 1],
        ("test",): [1, 1, 1],
        ("validation",): [1, 1, 1],
    }
    prepared_data = prepare_data(dataset=dataset, metric=partial(label_counter, field="entities"))
    assert prepared_data == {
        ("train", "ORG"): [2],
        ("train", "MISC"): [3],
        ("train", "PER"): [2],
        ("train", "LOC"): [2],
        ("test", "LOC"): [2, 3],
        ("test", "PER"): [2, 2],
        ("validation", "ORG"): [2, 3],
        ("validation", "LOC"): [2],
        ("validation", "MISC"): [2],
        ("validation", "PER"): [2],
    }
    prepared_data = prepare_data(
        dataset=dataset, metric=partial(document_field_length_collector, field="text")
    )
    assert prepared_data == {
        ("train",): [48, 15, 19],
        ("test",): [57, 11, 40],
        ("validation",): [65, 17, 187],
    }


def test_prepare_data_tokenize(dataset):
    prepared_data = prepare_data(
        dataset=dataset,
        metric=DocumentTokenCounter(
            field="text", tokenizer_name_or_path="bert-base-uncased", add_special_tokens=False
        ),
    )
    assert prepared_data == {
        ("train",): [9, 2, 6],
        ("test",): [12, 4, 12],
        ("validation",): [11, 6, 38],
    }
