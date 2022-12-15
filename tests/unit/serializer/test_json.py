import json
from dataclasses import dataclass

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

from src.serializer import JsonSerializer


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


def test_serialize_json(tmp_path):
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

    path = str(tmp_path / "out.jsonl")
    serializer = JsonSerializer(path=path)

    serializer(documents=[document])

    loaded = [json.loads(line) for line in open(path).readlines()]
    document_loaded = ExampleDocument.fromdict(loaded[0])
    assert document_loaded == document
