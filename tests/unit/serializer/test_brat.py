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
def document():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin. He works at DFKI.", id="tmp_1"
    )

    document.labeled_spans.predictions.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),  # Harry
            LabeledSpan(start=15, end=21, label="LOCATION"),  # Berlin
        ]
    )

    document.labeled_spans.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),  # Harry
            LabeledSpan(start=15, end=21, label="LOCATION"),  # Berlin
            LabeledSpan(start=35, end=39, label="ORGANIZATION"),  # DFKI
        ]
    )

    document.binary_relations.predictions.extend(
        [
            BinaryRelation(
                head=document.labeled_spans[0], tail=document.labeled_spans[1], label="lives_in"
            ),
        ]
    )
    document.binary_relations.extend(
        [
            BinaryRelation(
                head=document.labeled_spans[0], tail=document.labeled_spans[1], label="lives_in"
            ),
            BinaryRelation(
                head=document.labeled_spans[0], tail=document.labeled_spans[2], label="works_at"
            ),
        ]
    )

    return document


def test_save(tmp_path, document, document_processor):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        document_processor=document_processor,
        entity_layer="spans",
        relation_layer="relations",
    )

    metadata = serializer(documents=[document])

    path = metadata["path"]
    res = read_annotation_file(os.path.join(path, f"{document.id}.ann"))
    """
        res in the following format:
        {'spans':
            [
                {'id': 'T0', 'text': 'Harry', 'type': 'PERSON', 'locations': [{'start': 0, 'end': 5}]},
                {'id': 'T1', 'text': 'DFKI', 'type': 'ORGANIZATION', 'locations': [{'start': 35, 'end': 39}]},
                {'id': 'T2', 'text': 'Berlin', 'type': 'LOCATION', 'locations': [{'start': 15, 'end': 21}]}
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
    original_spans = document.labeled_spans.predictions
    assert len(spans) == len(original_spans)

    sorted_spans = sorted(
        spans, key=lambda x: x["locations"][0]["start"]
    )  # sort by start index of first span
    sorted_original_spans = sorted(original_spans, key=lambda x: x.start)  # sort by start index
    spanid2span = dd()  # map span_id (T0,T1,..) to original span
    for span, original_span in zip(sorted_spans, sorted_original_spans):
        assert span["locations"][0]["start"] == original_span.start
        assert span["locations"][0]["end"] == original_span.end
        assert span["type"] == original_span.label
        assert span["text"] == document.text[original_span.start : original_span.end]

        spanid2span[span["id"]] = original_span

    relations = res["relations"]
    original_relations = document.binary_relations.predictions
    assert len(relations) == len(original_relations)

    sorted_relations = sorted(relations, key=lambda x: x["type"])  # sort by relation label
    sorted_original_relations = sorted(original_relations, key=lambda x: x.label)

    for relation, original_relation in zip(sorted_relations, sorted_original_relations):
        assert relation["type"] == original_relation.label
        assert spanid2span[relation["arguments"][0]["target"]] == original_relation.head
        assert spanid2span[relation["arguments"][1]["target"]] == original_relation.tail


def test_save_gold_annotation_with_prefix(tmp_path, document, document_processor):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        document_processor=document_processor,
        entity_layer="spans",
        relation_layer="relations",
        gold_label_prefix="GOLD",
    )

    metadata = serializer(documents=[document])

    path = metadata["path"]
    res = read_annotation_file(os.path.join(path, f"{document.id}.ann"))
    """
    res in the following format:
    {
        'spans':
            [
                {'id': 'T0', 'text': 'Berlin', 'type': 'GOLD-LOCATION', 'locations': [{'start': 15, 'end': 21}]},
                {'id': 'T1', 'text': 'Harry', 'type': 'GOLD-PERSON', 'locations': [{'start': 0, 'end': 5}]},
                {'id': 'T2', 'text': 'DFKI', 'type': 'GOLD-ORGANIZATION', 'locations': [{'start': 35, 'end': 39}]},
                {'id': 'T3', 'text': 'Berlin', 'type': 'LOCATION', 'locations': [{'start': 15, 'end': 21}]},
                {'id': 'T4', 'text': 'Harry', 'type': 'PERSON', 'locations': [{'start': 0, 'end': 5}]}
            ],
        'relations':
            [
                {'id': 'R0', 'type': 'GOLD-lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T1'},
                    {'type': 'Arg2', 'target': 'T0'}]},
                {'id': 'R1', 'type': 'GOLD-works_at', 'arguments': [{'type': 'Arg1', 'target': 'T1'},
                    {'type': 'Arg2', 'target': 'T2'}]},
                {'id': 'R3', 'type': 'lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T4'},
                    {'type': 'Arg2', 'target': 'T3'}]}
            ]
    }

    """
    spans = res["spans"]
    sorted_spans = sorted(spans, key=lambda x: x["type"])[
        :3
    ]  # only first three are gold annotations
    original_spans = document.labeled_spans
    sorted_original_spans = sorted(original_spans, key=lambda x: x.label)
    assert len(sorted_spans) == len(sorted_original_spans)

    spanid2span = dd()
    for span, original_span in zip(sorted_spans, sorted_original_spans):
        assert span["locations"][0]["start"] == original_span.start
        assert span["locations"][0]["end"] == original_span.end
        assert span["type"] == f"GOLD-{original_span.label}"
        assert span["text"] == document.text[original_span.start : original_span.end]

        spanid2span[span["id"]] = original_span

    relations = res["relations"]
    sorted_relations = sorted(relations, key=lambda x: x["type"])[
        :2
    ]  # only first two are gold annotations
    original_relations = document.binary_relations
    sorted_original_relations = sorted(original_relations, key=lambda x: x.label)
    assert len(sorted_relations) == len(sorted_original_relations)

    for relation, original_relation in zip(sorted_relations, sorted_original_relations):
        assert relation["type"] == f"GOLD-{original_relation.label}"
        assert spanid2span[relation["arguments"][0]["target"]] == original_relation.head
        assert spanid2span[relation["arguments"][1]["target"]] == original_relation.tail


def test_save_gold_and_predicted_annotation_with_prefix(tmp_path, document, document_processor):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        document_processor=document_processor,
        entity_layer="spans",
        relation_layer="relations",
        gold_label_prefix="GOLD",
        prediction_label_prefix="PRED",
    )

    metadata = serializer(documents=[document])

    path = metadata["path"]
    res = read_annotation_file(os.path.join(path, f"{document.id}.ann"))
    """
    res in the following format:
    {
        'spans':
            [
                {'id': 'T0', 'text': 'Berlin', 'type': 'GOLD-LOCATION', 'locations': [{'start': 15, 'end': 21}]},
                {'id': 'T1', 'text': 'Harry', 'type': 'GOLD-PERSON', 'locations': [{'start': 0, 'end': 5}]},
                {'id': 'T2', 'text': 'DFKI', 'type': 'GOLD-ORGANIZATION', 'locations': [{'start': 35, 'end': 39}]},
                {'id': 'T3', 'text': 'Berlin', 'type': 'PRED-LOCATION', 'locations': [{'start': 15, 'end': 21}]},
                {'id': 'T4', 'text': 'Harry', 'type': 'PRED-PERSON', 'locations': [{'start': 0, 'end': 5}]}
            ],
        'relations':
            [
                {'id': 'R0', 'type': 'GOLD-lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T1'},
                    {'type': 'Arg2', 'target': 'T0'}]},
                {'id': 'R1', 'type': 'GOLD-works_at', 'arguments': [{'type': 'Arg1', 'target': 'T1'},
                    {'type': 'Arg2', 'target': 'T2'}]},
                {'id': 'R3', 'type': 'PRED-lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T4'},
                    {'type': 'Arg2', 'target': 'T3'}]}
            ]
    }

    """
    spans = res["spans"]
    sorted_spans = sorted(spans, key=lambda x: x["type"])

    gold_spans = sorted_spans[:3]  # only first three are gold annotations
    original_gold_spans = document.labeled_spans
    sorted_original_gold_spans = sorted(original_gold_spans, key=lambda x: x.label)
    assert len(gold_spans) == len(sorted_original_gold_spans)

    spanid2span = dd()
    for span, original_span in zip(gold_spans, sorted_original_gold_spans):
        assert span["locations"][0]["start"] == original_span.start
        assert span["locations"][0]["end"] == original_span.end
        assert span["type"] == f"GOLD-{original_span.label}"
        assert span["text"] == document.text[original_span.start : original_span.end]

        spanid2span[span["id"]] = original_span

    predicted_spans = sorted_spans[3:]  # last two are predicted annotations
    original_predicted_spans = document.labeled_spans.predictions
    sorted_original_predicted_spans = sorted(original_predicted_spans, key=lambda x: x.label)
    assert len(predicted_spans) == len(sorted_original_predicted_spans)

    for span, original_span in zip(predicted_spans, sorted_original_predicted_spans):
        assert span["locations"][0]["start"] == original_span.start
        assert span["locations"][0]["end"] == original_span.end
        assert span["type"] == f"PRED-{original_span.label}"
        assert span["text"] == document.text[original_span.start : original_span.end]

        spanid2span[span["id"]] = original_span

    relations = res["relations"]
    sorted_relations = sorted(relations, key=lambda x: x["type"])

    gold_relations = sorted_relations[:2]  # only first two are gold annotations
    original_gold_relations = document.binary_relations
    sorted_original_relations = sorted(original_gold_relations, key=lambda x: x.label)
    assert len(gold_relations) == len(sorted_original_relations)

    for relation, original_relation in zip(gold_relations, sorted_original_relations):
        assert relation["type"] == f"GOLD-{original_relation.label}"
        assert spanid2span[relation["arguments"][0]["target"]] == original_relation.head
        assert spanid2span[relation["arguments"][1]["target"]] == original_relation.tail

    predicted_relations = sorted_relations[2:]  # only last annotation is predicted annotation
    original_predicted_relations = document.binary_relations.predictions
    sorted_original_relations = sorted(original_predicted_relations, key=lambda x: x.label)
    assert len(predicted_relations) == len(sorted_original_relations)

    for relation, original_relation in zip(predicted_relations, sorted_original_relations):
        assert relation["type"] == f"PRED-{original_relation.label}"
        assert spanid2span[relation["arguments"][0]["target"]] == original_relation.head
        assert spanid2span[relation["arguments"][1]["target"]] == original_relation.tail


@pytest.fixture
def document_with_multispan():
    document = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )
    document.labeled_spans.predictions.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=15, end=21, label="LOCATION"),
            LabeledSpan(start=23, end=30, label="LOCATION"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )
    # add relations as predictions

    document.binary_relations.predictions.extend(
        [
            BinaryRelation(
                head=document.labeled_spans.predictions[0],
                tail=document.labeled_spans.predictions[1],
                label="lives_in",
            ),
            BinaryRelation(
                head=document.labeled_spans.predictions[1],
                tail=document.labeled_spans.predictions[2],
                label="parts_of_same",
            ),  # should be removed
            BinaryRelation(
                head=document.labeled_spans.predictions[1],
                tail=document.labeled_spans.predictions[3],
                label="parts_of_same",
            ),  # should be removed
            BinaryRelation(
                head=document.labeled_spans.predictions[0],
                tail=document.labeled_spans.predictions[3],
                label="works_at",
            ),
            BinaryRelation(
                head=document.labeled_spans.predictions[3],
                tail=document.labeled_spans.predictions[1],
                label="located_in",
            ),  # tail should be a new merged entity
            BinaryRelation(
                head=document.labeled_spans.predictions[3],
                tail=document.labeled_spans.predictions[2],
                label="located_in",
            ),  # tail should be a new merged entity
        ]
    )

    return document


def test_save_multispan(tmp_path, document_with_multispan, document_processor):
    path = str(tmp_path)
    serializer = BratSerializer(
        path=path,
        document_processor=document_processor,
        entity_layer="spans",
        relation_layer="relations",
    )

    metadata = serializer(documents=[document_with_multispan])

    path = metadata["path"]
    res = read_annotation_file(os.path.join(path, f"{document_with_multispan.id}.ann"))

    """
    {
        'spans':
            [
                {'id': 'T0', 'text': 'DFKI', 'type': 'ORGANIZATION', 'locations': [{'start': 44, 'end': 48}]},
                {'id': 'T1', 'text': 'Harry', 'type': 'PERSON', 'locations': [{'start': 0, 'end': 5}]},
                {'id': 'T2', 'text': 'Berlin Germany', 'type': 'LOCATION',
                    'locations': [{'start': 15, 'end': 21}, {'start': 23, 'end': 30}]}
            ],
        'relations':
            [
                {'id': 'R0', 'type': 'lives_in', 'arguments': [{'type': 'Arg1', 'target': 'T1'},
                    {'type': 'Arg2', 'target': 'T2'}]},
                {'id': 'R1', 'type': 'located_in', 'arguments': [{'type': 'Arg1', 'target': 'T0'},
                    {'type': 'Arg2', 'target': 'T2'}]},
                {'id': 'R2', 'type': 'works_at', 'arguments': [{'type': 'Arg1', 'target': 'T1'},
                    {'type': 'Arg2', 'target': 'T0'}]}
            ]
    }

    """
    spans = res["spans"]
    assert len(spans) == 3

    sorted_spans = sorted(spans, key=lambda x: x["locations"][0]["start"])

    spanid2span = dd()

    span = sorted_spans[0]  # verify first span
    assert span["locations"][0]["start"] == 0
    assert span["locations"][0]["end"] == 5
    assert span["type"] == "PERSON"
    assert span["text"] == "Harry"
    spanid2span[span["id"]] = span

    span = sorted_spans[1]  # verify second span (multispan)
    assert span["locations"][0]["start"] == 15
    assert span["locations"][0]["end"] == 21
    assert span["locations"][1]["start"] == 23
    assert span["locations"][1]["end"] == 30
    assert span["type"] == "LOCATION"
    assert span["text"] == "Berlin Germany"
    spanid2span[span["id"]] = span

    relations = res["relations"]
    assert len(relations) == 3

    sorted_relations = sorted(relations, key=lambda x: x["type"])

    relation = sorted_relations[0]  # verify relation between first and second span
    relation_type = relation["type"]
    assert relation_type == "lives_in"
    arg1 = spanid2span[relation["arguments"][0]["target"]]["text"]
    arg2 = spanid2span[relation["arguments"][1]["target"]]["text"]
    assert f"{arg1} {relation_type} {arg2}" == "Harry lives_in Berlin Germany"
