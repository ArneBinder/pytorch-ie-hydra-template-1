import dataclasses
import os
from dataclasses import dataclass
from typing import TypeVar

import pytest
from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TokenDocumentWithLabeledSpansAndBinaryRelations,
)
from pytorch_ie import AnnotationLayer
from pytorch_ie.core import Document, annotation_field

from src.serializer import BratSerializer
from src.serializer.brat import (
    serialize_annotations,
    serialize_binary_relation,
    serialize_labeled_span,
)
from src.utils import get_pylogger

log = get_pylogger(__name__)
D = TypeVar("D", bound=Document)


def test_serialize_labeled_span():
    # labeled span
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )
    document.labeled_spans.extend(
        [
            LabeledSpan(start=15, end=30, label="LOCATION"),
        ]
    )
    labeled_span = document.labeled_spans[0]
    serialized_labeled_span = serialize_labeled_span(
        span_id="T1",
        labeled_span=labeled_span,
    )
    assert serialized_labeled_span == "T1\tLOCATION 15 30\tBerlin, Germany\n"

    # labeled multi span
    @dataclasses.dataclass
    class TextDocumentWithLabeledMultiSpansAndBinaryRelations(TextBasedDocument):
        labeled_multi_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
        binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(
            target="labeled_multi_spans"
        )

    document = TextDocumentWithLabeledMultiSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )
    document.labeled_multi_spans.extend(
        [
            LabeledMultiSpan(slices=((15, 21), (23, 30)), label="LOCATION"),
        ]
    )
    labeled_multi_span = document.labeled_multi_spans[0]
    serialized_labeled_multi_span = serialize_labeled_span(
        span_id="T2",
        labeled_span=labeled_multi_span,
    )
    assert serialized_labeled_multi_span == "T2\tLOCATION 15 21;23 30\tBerlin Germany\n"

    # Unknown type
    binary_relation = BinaryRelation(
        head=document.labeled_multi_spans[0],
        tail=document.labeled_multi_spans[0],
        label="self",
    )
    with pytest.raises(Warning) as w:
        serialize_labeled_span(
            span_id="T0",
            labeled_span=binary_relation,
        )
    assert (
        str(w.value)
        == "labeled span has unknown type: <class 'pytorch_ie.annotations.BinaryRelation'>"
    )


def test_serialize_binary_relation():
    binary_relation = BinaryRelation(
        head=LabeledSpan(start=0, end=5, label="PERSON"),
        tail=LabeledSpan(start=15, end=30, label="LOCATION"),
        label="lives_in",
    )
    span2id = {binary_relation.head: "T1", binary_relation.tail: "T2"}
    serialized_binary_relation = serialize_binary_relation(
        relation_id="R1",
        binary_relation=binary_relation,
        span2id=span2id,
    )

    assert serialized_binary_relation == "R1\tlives_in Arg1:T1 Arg2:T2\n"

    # Unknown relation type
    with pytest.raises(Warning) as w:
        serialize_binary_relation(relation_id=0, binary_relation=None, span2id={})
    assert str(w.value) == "relation has unknown type: <class 'NoneType'>"


@dataclass
class TextDocumentWithLabeledMultiSpansAndBinaryRelations(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@pytest.fixture
def document():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )
    document.labeled_spans.predictions.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )

    assert str(document.labeled_spans.predictions[0]) == "Harry"
    assert str(document.labeled_spans.predictions[1]) == "DFKI"

    document.labeled_spans.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=15, end=30, label="LOCATION"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )
    assert str(document.labeled_spans[0]) == "Harry"
    assert str(document.labeled_spans[1]) == "Berlin, Germany"
    assert str(document.labeled_spans[2]) == "DFKI"

    document.binary_relations.predictions.extend(
        [
            BinaryRelation(
                head=document.labeled_spans[0],
                tail=document.labeled_spans[2],
                label="works_at",
            ),
        ]
    )

    document.binary_relations.extend(
        [
            BinaryRelation(
                head=document.labeled_spans[0],
                tail=document.labeled_spans[1],
                label="lives_in",
            ),
            BinaryRelation(
                head=document.labeled_spans[0],
                tail=document.labeled_spans[2],
                label="works_at",
            ),
        ]
    )

    return document


@pytest.fixture
def serialized_annotations(
    document,
    gold_label_prefix=None,
    prediction_label_prefix=None,
):
    return serialize_annotations(
        labeled_span_layer=document.labeled_spans,
        binary_relation_layer=document.binary_relations,
        gold_label_prefix=gold_label_prefix,
        prediction_label_prefix=prediction_label_prefix,
    )


@pytest.mark.parametrize(
    "gold_label_prefix, prediction_label_prefix",
    [(None, None), ("GOLD", None), (None, "PRED"), ("GOLD", "PRED")],
)
def test_serialize_annotations(document, gold_label_prefix, prediction_label_prefix):
    serialized_annotations = serialize_annotations(
        labeled_span_layer=document.labeled_spans,
        binary_relation_layer=document.binary_relations,
        gold_label_prefix=gold_label_prefix,
        prediction_label_prefix=prediction_label_prefix,
    )

    if gold_label_prefix == "GOLD" and prediction_label_prefix == "PRED":
        assert len(serialized_annotations) == 8
        assert serialized_annotations == [
            "T0\tGOLD-PERSON 0 5\tHarry\n",
            "T1\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T2\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "T3\tPRED-PERSON 0 5\tHarry\n",
            "T4\tPRED-ORGANIZATION 44 48\tDFKI\n",
            "R0\tGOLD-lives_in Arg1:T0 Arg2:T1\n",
            "R1\tGOLD-works_at Arg1:T0 Arg2:T2\n",
            "R2\tPRED-works_at Arg1:T3 Arg2:T4\n",
        ]
    elif gold_label_prefix == "GOLD" and prediction_label_prefix is None:
        assert len(serialized_annotations) == 8
        assert serialized_annotations == [
            "T0\tGOLD-PERSON 0 5\tHarry\n",
            "T1\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T2\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "T3\tPERSON 0 5\tHarry\n",
            "T4\tORGANIZATION 44 48\tDFKI\n",
            "R0\tGOLD-lives_in Arg1:T0 Arg2:T1\n",
            "R1\tGOLD-works_at Arg1:T0 Arg2:T2\n",
            "R2\tworks_at Arg1:T3 Arg2:T4\n",
        ]
    elif gold_label_prefix is None and prediction_label_prefix == "PRED":
        assert len(serialized_annotations) == 3
        assert serialized_annotations == [
            "T0\tPRED-PERSON 0 5\tHarry\n",
            "T1\tPRED-ORGANIZATION 44 48\tDFKI\n",
            "R0\tPRED-works_at Arg1:T0 Arg2:T1\n",
        ]
    else:
        assert len(serialized_annotations) == 3
        assert serialized_annotations == [
            "T0\tPERSON 0 5\tHarry\n",
            "T1\tORGANIZATION 44 48\tDFKI\n",
            "R0\tworks_at Arg1:T0 Arg2:T1\n",
        ]


def document_processor(document) -> TextBasedDocument:
    doc = document.copy()
    doc["labeled_spans"].append(LabeledSpan(start=0, end=0, label="empty"))
    return doc


def test_write(tmp_path, document, serialized_annotations):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        document_processor=document_processor,
        entity_layer="labeled_spans",
        relation_layer="binary_relations",
    )

    metadata = serializer(documents=[document])
    path = metadata["path"]
    ann_file = os.path.join(path, f"{document.id}.ann")

    with open(ann_file, "r") as file:
        for i, line in enumerate(file.readlines()):
            assert line == serialized_annotations[i]
    file.close()


@pytest.fixture
def dummy_document():
    document = TokenDocumentWithLabeledSpansAndBinaryRelations(
        id="tmp_1",
        tokens=(),
    )
    return document


def test_write_with_exceptions_and_warnings(tmp_path, caplog, document, dummy_document):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        entity_layer="labeled_spans",
        relation_layer="binary_relations",
    )

    # list of empty documents
    with pytest.raises(Exception) as e:
        serializer(documents=[])
    assert str(e.value) == "cannot serialize empty list of documents"

    # List of documents with type unexpected Document type
    with pytest.raises(TypeError) as type_error:
        serializer(documents=[dummy_document])
    assert str(type_error.value) == (
        "Document tmp_1 has unexpected type: <class "
        "'pie_modules.documents.TokenDocumentWithLabeledSpansAndBinaryRelations'>. BratSerializer "
        "can only serialize TextBasedDocuments."
    )

    # Warning when metadata file already exists
    metadata = serializer(documents=[document])
    full_metadata_file_name = os.path.join(metadata["path"], metadata["metadata_file_name"])
    serializer(documents=[document])

    assert caplog.records[0].levelname == "WARNING"
    assert (
        f"metadata file {full_metadata_file_name} already exists, it will be overwritten!\n"
        in caplog.text
    )


@pytest.mark.parametrize("split", [None, "test"])
def test_write_with_split(tmp_path, document, split):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path, entity_layer="labeled_spans", relation_layer="binary_relations", split=split
    )

    metadata = serializer(documents=[document])
    real_path = metadata["path"]
    if split is None:
        assert real_path == os.path.join(path)
    elif split is not None:
        assert real_path == os.path.join(path, split)
