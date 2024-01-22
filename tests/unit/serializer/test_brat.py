from dataclasses import dataclass
from typing import TypeVar

from pytorch_ie.core import Document

from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)
import pytest
from pie_datasets import DatasetDict
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

from src.serializer import BratSerializer


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def document():
    document = ExampleDocument(text="Harry lives in Berlin. He works at DFKI.", id="tmp")
    entities = [
        LabeledSpan(start=0, end=5, label="PERSON"),
        LabeledSpan(start=15, end=21, label="LOCATION"),
        LabeledSpan(start=35, end=39, label="ORGANIZATION"),
    ]
    for ent in entities:
        document.entities.predictions.append(ent)

    relations = [
        BinaryRelation(head=entities[0], tail=entities[1], label="lives_in"),
        BinaryRelation(head=entities[0], tail=entities[2], label="works_at"),
    ]

    # add relations as predictions
    for rel in relations:
        document.relations.predictions.append(rel)

    return document


@pytest.fixture
def document_with_multispan():
    document = ExampleDocument(text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp")
    entities = [
        LabeledSpan(start=0, end=5, label="PERSON"),
        LabeledSpan(start=15, end=21, label="LOCATION"),
        LabeledSpan(start=23, end=30, label="LOCATION"),
        LabeledSpan(start=44, end=48, label="ORGANIZATION"),
    ]
    for ent in entities:
        document.entities.predictions.append(ent)

    relations = [
        BinaryRelation(head=entities[0], tail=entities[1], label="lives_in"),
        BinaryRelation(head=entities[1], tail=entities[2], label="part_of_same"),
        BinaryRelation(head=entities[0], tail=entities[3], label="works_at"),
        # TODO: can have one or both relations below ?
        # BinaryRelation(head=entities[3], tail=entities[1], label="located_in"),
        # BinaryRelation(head=entities[3], tail=entities[2], label="located_in"),
    ]

    # add relations as predictions
    for rel in relations:
        document.relations.predictions.append(rel)

    return document


def test_save_and_load(tmp_path, document):
    path = str(tmp_path)
    serializer = BratSerializer(path=path)

    serializer(documents=[document])

    loaded_document = serializer.read_with_defaults()[0]
    assert loaded_document.text == document.text

    entities = document.entities.predictions
    loaded_entities = loaded_document.entities.predictions
    assert loaded_entities == entities

    relations = document.relations.predictions
    loaded_relations = loaded_document.relations.predictions
    assert loaded_relations == relations


def test_save_and_load_multispan(tmp_path, document_with_multispan):
    path = str(tmp_path)
    serializer = BratSerializer(path=path)

    serializer(documents=[document_with_multispan])

    loaded_document = serializer.read_with_defaults()[0]
    assert loaded_document.text == document_with_multispan.text

    entities = document_with_multispan.entities.predictions
    loaded_entities = loaded_document.entities.predictions
    assert loaded_entities == entities

    relations = document_with_multispan.relations.predictions
    loaded_relations = loaded_document.relations.predictions
    assert loaded_relations == relations
