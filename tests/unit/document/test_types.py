from pytorch_ie.annotations import LabeledSpan

from src.document.types import Attribute, TextDocumentWithLabeledEntitiesAndEntityAttributes


def test_attribute():
    entity = LabeledSpan(start=0, end=1, label="PER")
    attribute = Attribute(annotation=entity, label="FACT")

    assert attribute.annotation == entity
    assert attribute.label == "FACT"
    assert attribute.type is None
    assert attribute.score is None

    assert str(attribute) == "Attribute(label=FACT)"


def test_document():
    doc = TextDocumentWithLabeledEntitiesAndEntityAttributes("He is really a person.")
    entity = LabeledSpan(start=0, end=2, label="PER")
    doc.entities.append(entity)
    attribute = Attribute(annotation=entity, label="FACT")
    doc.entity_attributes.append(attribute)

    assert str(doc.entities[0]) == "He"
    assert str(doc.entity_attributes[0]) == "Attribute(label=FACT,annotation=He)"
