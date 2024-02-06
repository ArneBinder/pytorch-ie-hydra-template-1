import os
from collections import defaultdict as dd
from dataclasses import dataclass
from typing import TypeVar

from pie_datasets.builders.brat import BratDocument
from pie_modules.document.processing import SpansViaRelationMerger
from pytorch_ie import AnnotationLayer
from pytorch_ie.core import Document

from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)
import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import annotation_field
from pytorch_ie.documents import TextBasedDocument, TextDocumentWithLabeledSpansAndBinaryRelations

from src.serializer import BratSerializer


def get_location(location_string):
    parts = location_string.split(" ")
    assert (
        len(parts) == 2
    ), f"Wrong number of entries in location string. Expected 2, but found: {parts}"
    return {"start": int(parts[0]), "end": int(parts[1])}


def get_span_annotation(annotation_line):
    """example input:

    T1  Organization 0 4        Sony
    """

    _id, remaining, text = annotation_line.split("\t", maxsplit=2)
    _type, locations = remaining.split(" ", maxsplit=1)
    return {
        "id": _id,
        "text": text,
        "type": _type,
        "locations": [get_location(loc) for loc in locations.split(";")],
    }


def get_relation_annotation(annotation_line):
    """example input:

    R1  Origin Arg1:T3 Arg2:T4
    """

    _id, remaining = annotation_line.strip().split("\t")
    _type, remaining = remaining.split(" ", maxsplit=1)
    args = [dict(zip(["type", "target"], a.split(":"))) for a in remaining.split(" ")]
    return {"id": _id, "type": _type, "arguments": args}


def read_annotation_file(filename):
    res = {
        "spans": [],
        "relations": [],
    }
    with open(filename, encoding="utf-8") as file:
        for i, line in enumerate(file):
            if len(line.strip()) == 0:
                continue
            ann_type = line[0]
            # strip away the new line character
            if line.endswith("\n"):
                line = line[:-1]
            if ann_type == "T":
                res["spans"].append(get_span_annotation(line))
            elif ann_type == "R":
                res["relations"].append(get_relation_annotation(line))
            else:
                raise ValueError(
                    f'unknown BRAT annotation id type: "{line}" (from file {filename} @line {i}). '
                    f"Annotation ids have to start with T (spans), E (events), R (relations), "
                    f"A (attributions), or N (normalizations). See "
                    f"https://brat.nlplab.org/standoff.html for the BRAT annotation file "
                    f"specification."
                )
    return res


@dataclass
class TextDocumentWithLabeledMultiSpansAndBinaryRelations(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@pytest.fixture
def document_processor():
    dp = SpansViaRelationMerger(
        relation_layer="binary_relations",
        link_relation_label="parts_of_same",
        result_document_type=BratDocument,
        result_field_mapping={"labeled_spans": "spans", "binary_relations": "relations"},
    )

    return dp


@pytest.fixture
def document_with_gold_only():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin. He works at DFKI.", id="tmp"
    )
    labeled_spans = [
        LabeledSpan(start=0, end=5, label="PERSON"),
        LabeledSpan(start=15, end=21, label="LOCATION"),
        LabeledSpan(start=35, end=39, label="ORGANIZATION"),
    ]
    for ent in labeled_spans:
        document.labeled_spans.append(ent)

    binary_relations = [
        BinaryRelation(head=labeled_spans[0], tail=labeled_spans[1], label="lives_in"),
        BinaryRelation(head=labeled_spans[0], tail=labeled_spans[2], label="works_at"),
    ]

    for rel in binary_relations:
        document.binary_relations.append(rel)

    return document


def test_save_gold_only(tmp_path, document_with_gold_only, document_processor):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path, document_processor=document_processor, gold_label_prefix="GOLD"
    )

    metadata = serializer(documents=[document_with_gold_only])

    path = metadata["path"]
    res = read_annotation_file(os.path.join(path, f"{document_with_gold_only.id}.ann"))
    """
    res in the following format:
    {'spans':
        [
            {'id': 'T0', 'text': 'Harry', 'type': 'GOLD-PERSON', 'locations': [{'start': 0, 'end': 5}]},
            {'id': 'T1', 'text': 'DFKI', 'type': 'GOLD-ORGANIZATION', 'locations': [{'start': 35, 'end': 39}]},
            {'id': 'T2', 'text': 'Berlin', 'type': 'GOLD-LOCATION', 'locations': [{'start': 15, 'end': 21}]}
        ],
    'relations':
        [
            {'id': 'R0', 'type': 'lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T0'},
                {'type': 'Arg2', 'target': 'T2'}]},
            {'id': 'R1', 'type': 'works_at', 'arguments': [{'type': 'Arg1', 'target': 'T0'},
                {'type': 'Arg2', 'target': 'T1'}]}
        ]
    }
    """
    spans = res["spans"]
    original_spans = document_with_gold_only.labeled_spans
    assert len(spans) == len(original_spans)

    sorted_spans = sorted(spans, key=lambda x: x["locations"][0]["start"])
    sorted_original_spans = sorted(original_spans, key=lambda x: x.start)
    span2spanid = dd()
    for span, original_span in zip(sorted_spans, sorted_original_spans):
        assert span["locations"][0]["start"] == original_span.start
        assert span["locations"][0]["end"] == original_span.end
        assert span["type"] == f"GOLD-{original_span.label}"
        assert (
            span["text"] == document_with_gold_only.text[original_span.start : original_span.end]
        )

        span2spanid[original_span] = span["id"]

    relations = res["relations"]
    original_relations = document_with_gold_only.binary_relations
    assert len(relations) == len(original_relations)

    sorted_relations = sorted(relations, key=lambda x: x["type"])
    sorted_original_relations = sorted(original_relations, key=lambda x: x.label)

    for relation, original_relation in zip(sorted_relations, sorted_original_relations):
        assert relation["type"] == original_relation.label
        assert relation["arguments"][0]["target"] == span2spanid[original_relation.head]
        assert relation["arguments"][1]["target"] == span2spanid[original_relation.tail]


@pytest.fixture
def document_with_prediction_only():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin. He works at DFKI.", id="tmp"
    )
    labeled_spans = [
        LabeledSpan(start=0, end=5, label="PERSON"),
        LabeledSpan(start=15, end=21, label="LOCATION"),
        LabeledSpan(start=35, end=39, label="ORGANIZATION"),
    ]

    # add entities as predictions
    for ent in labeled_spans:
        document.labeled_spans.predictions.append(ent)

    binary_relations = [
        BinaryRelation(head=labeled_spans[0], tail=labeled_spans[1], label="lives_in"),
        BinaryRelation(head=labeled_spans[0], tail=labeled_spans[2], label="works_at"),
    ]

    # add relations as predictions
    for rel in binary_relations:
        document.binary_relations.predictions.append(rel)

    return document


def test_save_prediction_only(tmp_path, document_with_prediction_only, document_processor):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path, document_processor=document_processor, prediction_label_prefix="PRED"
    )

    metadata = serializer(documents=[document_with_prediction_only])

    path = metadata["path"]
    res = read_annotation_file(os.path.join(path, f"{document_with_prediction_only.id}.ann"))
    """
        res in the following format:
        {'spans':
            [
                {'id': 'T0', 'text': 'Harry', 'type': 'PRED-PERSON', 'locations': [{'start': 0, 'end': 5}]},
                {'id': 'T1', 'text': 'DFKI', 'type': 'PRED-ORGANIZATION', 'locations': [{'start': 35, 'end': 39}]},
                {'id': 'T2', 'text': 'Berlin', 'type': 'PRED-LOCATION', 'locations': [{'start': 15, 'end': 21}]}
            ],
        'relations':
            [
                {'id': 'R0', 'type': 'lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T0'},
                    {'type': 'Arg2', 'target': 'T2'}]},
                {'id': 'R1', 'type': 'works_at', 'arguments': [{'type': 'Arg1', 'target': 'T0'},
                    {'type': 'Arg2', 'target': 'T1'}]}
            ]
        }
        """
    spans = res["spans"]
    original_spans = document_with_prediction_only.labeled_spans.predictions
    assert len(spans) == len(original_spans)

    sorted_spans = sorted(spans, key=lambda x: x["locations"][0]["start"])
    sorted_original_spans = sorted(original_spans, key=lambda x: x.start)
    span2spanid = dd()
    for span, original_span in zip(sorted_spans, sorted_original_spans):
        assert span["locations"][0]["start"] == original_span.start
        assert span["locations"][0]["end"] == original_span.end
        assert span["type"] == f"PRED-{original_span.label}"
        assert (
            span["text"]
            == document_with_prediction_only.text[original_span.start : original_span.end]
        )

        span2spanid[original_span] = span["id"]

    relations = res["relations"]
    original_relations = document_with_prediction_only.binary_relations.predictions
    assert len(relations) == len(original_relations)

    sorted_relations = sorted(relations, key=lambda x: x["type"])
    sorted_original_relations = sorted(original_relations, key=lambda x: x.label)

    for relation, original_relation in zip(sorted_relations, sorted_original_relations):
        assert relation["type"] == original_relation.label
        assert relation["arguments"][0]["target"] == span2spanid[original_relation.head]
        assert relation["arguments"][1]["target"] == span2spanid[original_relation.tail]
