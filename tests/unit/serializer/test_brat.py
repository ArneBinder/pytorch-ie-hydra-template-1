import dataclasses
import os
from dataclasses import dataclass

import pytest
from pytorch_ie import Annotation, AnnotationLayer, Document
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import annotation_field
from pytorch_ie.documents import TextBasedDocument

from src.serializer import BratSerializer
from src.serializer.brat import (
    serialize_annotation,
    serialize_annotation_layers,
    serialize_binary_relation,
)


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndBinaryRelations(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


def test_serialize_labeled_span():

    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )
    document.labeled_spans.extend(
        [
            LabeledSpan(start=15, end=30, label="LOCATION"),
        ]
    )
    labeled_span = document.labeled_spans[0]
    annotation_type, serialized_annotation = serialize_annotation(
        annotation=labeled_span,
        annotation2id={},
    )
    assert annotation_type == "T"
    assert serialized_annotation == "LOCATION 15 30\tBerlin, Germany\n"


def test_serialize_labeled_multi_span():
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
    annotation_type, serialized_annotation = serialize_annotation(
        annotation=labeled_multi_span,
        annotation2id={},
    )
    assert annotation_type == "T"
    assert serialized_annotation == "LOCATION 15 21;23 30\tBerlin Germany\n"


def test_serialize_binary_relation():
    binary_relation = BinaryRelation(
        head=LabeledSpan(start=0, end=5, label="PERSON"),
        tail=LabeledSpan(start=15, end=30, label="LOCATION"),
        label="lives_in",
    )
    span2id = {binary_relation.head: "T1", binary_relation.tail: "T2"}
    annotation_type, serialized_binary_relation = serialize_binary_relation(
        annotation=binary_relation,
        annotation2id=span2id,
    )
    assert annotation_type == "R"
    assert serialized_binary_relation == "lives_in Arg1:T1 Arg2:T2\n"


def test_serialize_unknown_annotation():

    @dataclasses.dataclass(frozen=True)
    class InvalidAnnotation(Annotation):
        pass

    with pytest.raises(Warning) as w:
        serialize_annotation(annotation=InvalidAnnotation(), annotation2id={})
    assert (
        str(w.value)
        == "annotation has unknown type: <class 'tests.unit.serializer.test_brat.test_serialize_unknown_annotation.<locals>.InvalidAnnotation'>"
    )


@dataclass
class TextDocumentWithLabeledMultiSpansAndBinaryRelations(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@pytest.fixture
def document():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.",
        id="tmp",
        metadata={
            "span_ids": [],
            "relation_ids": [],
            "prediction_span_ids": [],
            "prediction_relation_ids": [],
        },
    )
    document.labeled_spans.predictions.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )
    document.metadata["prediction_span_ids"].extend(["T200", "T201"])

    assert str(document.labeled_spans.predictions[0]) == "Harry"
    assert str(document.labeled_spans.predictions[1]) == "DFKI"

    document.labeled_spans.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=15, end=30, label="LOCATION"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )
    document.metadata["span_ids"].extend(["T100", "T101", "T102"])

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
    document.metadata["prediction_relation_ids"].extend(["R200"])

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
    document.metadata["relation_ids"].extend(["R100", "R101"])

    return document


@pytest.mark.parametrize(
    "what",
    ["gold", "prediction", "both"],
)
def test_serialize_annotations(document, what):

    serialized_annotations = serialize_annotation_layers(
        layers=[(document.labeled_spans, what), (document.binary_relations, what)],
        gold_label_prefix="GOLD",
        prediction_label_prefix="PRED" if what == "both" else None,
    )

    if what == "both":
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
    elif what == "gold":
        assert serialized_annotations == [
            "T0\tGOLD-PERSON 0 5\tHarry\n",
            "T1\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T2\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "R0\tGOLD-lives_in Arg1:T0 Arg2:T1\n",
            "R1\tGOLD-works_at Arg1:T0 Arg2:T2\n",
        ]
    elif what == "prediction":
        assert serialized_annotations == [
            "T0\tPERSON 0 5\tHarry\n",
            "T1\tORGANIZATION 44 48\tDFKI\n",
            "R0\tworks_at Arg1:T0 Arg2:T1\n",
        ]
    else:
        raise ValueError(f"Unexpected value for what: {what}")


@pytest.mark.parametrize(
    "what",
    ["gold", "prediction", "both"],
)
def test_serialize_annotations_with_annotation_ids(document, what):
    serialized_annotations = serialize_annotation_layers(
        layers=[(document.labeled_spans, what), (document.binary_relations, what)],
        gold_label_prefix="GOLD",
        prediction_label_prefix="PRED" if what == "both" else None,
        gold_annotation_ids=[document.metadata["span_ids"], document.metadata["relation_ids"]],
        prediction_annotation_ids=[
            document.metadata["prediction_span_ids"],
            document.metadata["prediction_relation_ids"],
        ],
    )

    if what == "both":
        assert serialized_annotations == [
            "T100\tGOLD-PERSON 0 5\tHarry\n",
            "T101\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T102\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "T200\tPRED-PERSON 0 5\tHarry\n",
            "T201\tPRED-ORGANIZATION 44 48\tDFKI\n",
            "R100\tGOLD-lives_in Arg1:T100 Arg2:T101\n",
            "R101\tGOLD-works_at Arg1:T100 Arg2:T102\n",
            "R200\tPRED-works_at Arg1:T200 Arg2:T201\n",
        ]
    elif what == "gold":
        assert serialized_annotations == [
            "T100\tGOLD-PERSON 0 5\tHarry\n",
            "T101\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T102\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "R100\tGOLD-lives_in Arg1:T100 Arg2:T101\n",
            "R101\tGOLD-works_at Arg1:T100 Arg2:T102\n",
        ]
    elif what == "prediction":
        assert serialized_annotations == [
            "T200\tPERSON 0 5\tHarry\n",
            "T201\tORGANIZATION 44 48\tDFKI\n",
            "R200\tworks_at Arg1:T200 Arg2:T201\n",
        ]
    else:
        raise ValueError(f"Unexpected value for what: {what}")


def test_serialize_annotations_unknown_what(document):
    with pytest.raises(ValueError) as e:
        serialize_annotation_layers(
            layers=[(document.labeled_spans, "dummy"), (document.binary_relations, "dummy")],
        )
    assert (
        str(e.value)
        == 'Invalid value for what to serialize: "dummy". Expected "gold", "prediction", or "both".'
    )


def test_serialize_annotations_missing_prefix(document):

    with pytest.raises(ValueError) as e:
        serialize_annotation_layers(
            layers=[(document.labeled_spans, "both")],
        )
    assert str(e.value) == (
        "Cannot serialize both gold and prediction annotations without a label prefix "
        "for at least one of them. Consider setting gold_label_prefix or prediction_label_prefix."
    )


def append_empty_span_to_labeled_spans(document) -> TextBasedDocument:
    doc = document.copy()
    doc["labeled_spans"].append(LabeledSpan(start=0, end=0, label="empty"))
    return doc


def test_write(tmp_path, document):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        document_processor=append_empty_span_to_labeled_spans,
        layers={"labeled_spans": "prediction", "binary_relations": "prediction"},
    )

    metadata = serializer(documents=[document])
    path = metadata["path"]
    ann_file = os.path.join(path, f"{document.id}.ann")

    with open(ann_file, "r") as file:
        lines = file.readlines()

    assert lines == [
        "T0\tPERSON 0 5\tHarry\n",
        "T1\tORGANIZATION 44 48\tDFKI\n",
        "R0\tworks_at Arg1:T0 Arg2:T1\n",
    ]


def test_write_with_annotation_ids(tmp_path, document):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        layers={"labeled_spans": "gold", "binary_relations": "prediction"},
        metadata_gold_id_keys={"labeled_spans": "span_ids"},
    )

    metadata = serializer(documents=[document])
    path = metadata["path"]
    ann_file = os.path.join(path, f"{document.id}.ann")

    with open(ann_file, "r") as file:
        lines = file.readlines()

    assert lines == [
        "T100\tPERSON 0 5\tHarry\n",
        "T101\tLOCATION 15 30\tBerlin, Germany\n",
        "T102\tORGANIZATION 44 48\tDFKI\n",
        "R0\tworks_at Arg1:T100 Arg2:T102\n",
    ]


def test_write_with_exceptions_and_warnings(tmp_path, caplog, document):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path, layers={"labeled_spans": "prediction", "binary_relations": "prediction"}
    )

    # list of empty documents
    with pytest.raises(Exception) as e:
        serializer(documents=[])
    assert str(e.value) == "cannot serialize empty list of documents"

    @dataclass
    class InvalidDocument(Document):
        pass

    # List of documents with type unexpected Document type
    with pytest.raises(TypeError) as type_error:
        serializer(documents=[InvalidDocument()])
    assert str(type_error.value) == (
        "Document doc_0 has unexpected type: <class 'tests.unit.serializer.test_brat.test_write_with_exceptions_and_warnings.<locals>.InvalidDocument'>. "
        "BratSerializer can only serialize TextBasedDocuments."
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
        path=path,
        layers={"labeled_spans": "prediction", "binary_relations": "prediction"},
        split=split,
    )

    metadata = serializer(documents=[document])
    real_path = metadata["path"]
    if split is None:
        assert real_path == os.path.join(path)
    elif split is not None:
        assert real_path == os.path.join(path, split)


@pytest.fixture
def document_only_gold_spans_both_relations():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.",
        id="tmp",
        metadata={
            "span_ids": [],
            "relation_ids": [],
            "prediction_span_ids": [],
            "prediction_relation_ids": [],
        },
    )

    document.labeled_spans.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=15, end=30, label="LOCATION"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )
    document.metadata["span_ids"].extend(["T100", "T101", "T102"])

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
    document.metadata["prediction_relation_ids"].extend(["R200"])

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
    document.metadata["relation_ids"].extend(["R100", "R101"])

    return document


@pytest.mark.parametrize(
    "what",
    [("gold", "prediction"), ("both", "prediction"), ("gold", "both"), ("both", "both")],
)
def test_serialize_annotations_only_gold_spans_both_relations(
    document_only_gold_spans_both_relations, what
):

    serialized_annotations = serialize_annotation_layers(
        layers=[
            (document_only_gold_spans_both_relations.labeled_spans, what[0]),
            (document_only_gold_spans_both_relations.binary_relations, what[1]),
        ],
        gold_label_prefix="GOLD",
        prediction_label_prefix="PRED",
    )
    if what in [("gold", "prediction"), ("both", "prediction")]:
        assert serialized_annotations == [
            "T0\tGOLD-PERSON 0 5\tHarry\n",
            "T1\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T2\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "R0\tPRED-works_at Arg1:T0 Arg2:T2\n",
        ]
    elif what in [("gold", "both"), ("both", "both")]:
        assert serialized_annotations == [
            "T0\tGOLD-PERSON 0 5\tHarry\n",
            "T1\tGOLD-LOCATION 15 30\tBerlin, Germany\n",
            "T2\tGOLD-ORGANIZATION 44 48\tDFKI\n",
            "R0\tGOLD-lives_in Arg1:T0 Arg2:T1\n",
            "R1\tGOLD-works_at Arg1:T0 Arg2:T2\n",
            "R2\tPRED-works_at Arg1:T0 Arg2:T2\n",
        ]
    else:
        raise ValueError(f"Unexpected value for what: {what}")
