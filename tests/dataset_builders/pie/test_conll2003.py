import datasets
import pytest
from pie_datasets import DatasetDict
from pytorch_ie.core import Document
from pytorch_ie.documents import TextDocumentWithLabeledSpans

from dataset_builders.pie.conll2003.conll2003 import Conll2003
from tests.dataset_builders import PIE_BASE_PATH

DATASET_NAME = "conll2003"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
HF_DATASET_PATH = Conll2003.BASE_DATASET_PATH
SPLIT_NAMES = {"train", "validation", "test"}
SPLIT_SIZES = {"train": 14041, "validation": 3250, "test": 3453}


@pytest.fixture(params=[config.name for config in Conll2003.BUILDER_CONFIGS], scope="module")
def dataset_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_dataset(dataset_name):
    return datasets.load_dataset(str(HF_DATASET_PATH), name=dataset_name)


def test_hf_dataset(hf_dataset):
    assert set(hf_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in hf_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def hf_example(hf_dataset):
    return hf_dataset["train"][0]


def test_hf_example(hf_example, dataset_name):
    if dataset_name == "conll2003":
        assert hf_example == {
            "chunk_tags": [11, 21, 11, 12, 21, 22, 11, 12, 0],
            "id": "0",
            "ner_tags": [3, 0, 7, 0, 0, 0, 7, 0, 0],
            "pos_tags": [22, 42, 16, 21, 35, 37, 16, 21, 7],
            "tokens": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
        }
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


@pytest.fixture(scope="module")
def document(hf_example, hf_dataset):
    conll2003 = Conll2003()
    generate_document_kwargs = conll2003._generate_document_kwargs(hf_dataset["train"])
    document = conll2003._generate_document(example=hf_example, **generate_document_kwargs)
    return document


def test_document(document, dataset_name):
    assert isinstance(document, Document)
    if dataset_name == "conll2003":
        assert document.text == "EU rejects German call to boycott British lamb ."
        entities = list(document.entities)
        assert len(entities) == 3
        assert str(entities[0]) == "EU"
        assert str(entities[1]) == "German"
        assert str(entities[2]) == "British"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


@pytest.fixture(scope="module")
def pie_dataset(dataset_name):
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH), name=dataset_name)


def test_pie_dataset(pie_dataset):
    assert set(pie_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in pie_dataset.items()}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module", params=list(Conll2003.DOCUMENT_CONVERTERS))
def converter_document_type(request):
    return request.param


@pytest.fixture(scope="module")
def converted_pie_dataset(pie_dataset, converter_document_type):
    pie_dataset_converted = pie_dataset.to_document_type(document_type=converter_document_type)
    return pie_dataset_converted


def test_converted_pie_dataset(converted_pie_dataset, converter_document_type):
    assert set(converted_pie_dataset) == SPLIT_NAMES
    split_sizes = {split_name: len(ds) for split_name, ds in converted_pie_dataset.items()}
    assert split_sizes == SPLIT_SIZES
    for ds in converted_pie_dataset.values():
        for document in ds:
            assert isinstance(document, converter_document_type)


@pytest.fixture(scope="module")
def converted_document(converted_pie_dataset):
    return converted_pie_dataset["train"][0]


def test_converted_document(converted_document, converter_document_type):
    assert isinstance(converted_document, converter_document_type)
    if converter_document_type == TextDocumentWithLabeledSpans:
        assert converted_document.text == "EU rejects German call to boycott British lamb ."
        entities = list(converted_document.labeled_spans)
        assert len(entities) == 3
        assert str(entities[0]) == "EU"
        assert str(entities[1]) == "German"
        assert str(entities[2]) == "British"
    else:
        raise ValueError(f"Unknown converter document type: {converter_document_type}")
