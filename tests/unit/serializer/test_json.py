from dataclasses import dataclass

import pytest
from pie_datasets import DatasetDict
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextBasedDocument

from src.serializer import JsonSerializer


@dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def document():
    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )
    entities = [
        LabeledSpan(start=65, end=75, label="PER"),
        LabeledSpan(start=96, end=100, label="ORG"),
        LabeledSpan(start=126, end=134, label="ORG"),
    ]
    for ent in entities:
        document.entities.append(ent)

    relations = [
        BinaryRelation(head=entities[0], tail=entities[1], label="per:employee_of"),
        BinaryRelation(head=entities[0], tail=entities[2], label="per:employee_of"),
    ]

    # add relations as predictions
    for rel in relations:
        document.relations.predictions.append(rel)

    return document


def test_save_and_load(tmp_path, document):
    path = str(tmp_path)
    serializer = JsonSerializer(path=path)

    serializer(documents=[document])

    loaded_document = serializer.read_with_defaults()[0]
    assert loaded_document == document


def test_save_and_load_with_dataset_dict(tmp_path, document):
    path = str(tmp_path)
    serializer = JsonSerializer(path=path)

    serializer(documents=[document], split="train")
    serializer(documents=[document], split="test", append=True)

    loaded_dataset = DatasetDict.from_json(data_dir=path)
    assert set(loaded_dataset) == {"train", "test"}
    assert len(loaded_dataset["train"]) == 1
    assert len(loaded_dataset["test"]) == 1
    assert loaded_dataset["train"][0] == document
    assert loaded_dataset["test"][0] == document
