import dataclasses

import pytest
from pie_modules.annotations import BinaryRelation, LabeledSpan
from pie_modules.documents import TextBasedDocument
from pytorch_ie import AnnotationLayer
from pytorch_ie.core import annotation_field
from pytorch_ie.models import (
    TransformerSpanClassificationModel,
    TransformerTextClassificationModel,
)
from pytorch_ie.taskmodules import (
    TransformerRETextClassificationTaskModule,
    TransformerSpanClassificationTaskModule,
)

from src.pipeline.ner_re_pipeline import (
    NerRePipeline,
    add_annotations_from_other_documents,
    clear_annotation_layers,
    move_annotations_from_predictions,
    move_annotations_to_predictions,
    process_documents,
    process_pipeline_steps,
)


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndBinaryRelations(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
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

    return document


@pytest.fixture
def document_with_relations(document):

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


def test_clear_annotation_layers(document):
    new_doc = clear_annotation_layers(
        document,
        layer_names=["labeled_spans"],
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 2

    new_entities = new_doc["labeled_spans"]
    assert len(new_entities) == 0

    new_predictions = new_doc["labeled_spans"].predictions
    assert len(new_predictions) == 2

    # clear predictions

    new_doc = clear_annotation_layers(
        document,
        layer_names=["labeled_spans"],
        predictions=True,
    )

    new_entities = new_doc["labeled_spans"]
    assert len(new_entities) == 3

    new_predictions = new_doc["labeled_spans"].predictions
    assert len(new_predictions) == 0

    # clear inplace

    new_doc = clear_annotation_layers(
        document,
        layer_names=["labeled_spans"],
        predictions=True,
        inplace=True,
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 0

    new_doc = clear_annotation_layers(
        document,
        layer_names=["labeled_spans"],
        inplace=True,
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 0


def test_move_annotations_from_predictions(document):
    new_doc = move_annotations_from_predictions(
        document,
        layer_names=["labeled_spans"],
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 2

    new_entities = new_doc["labeled_spans"]
    assert len(new_entities) == 2

    new_predictions = new_doc["labeled_spans"].predictions
    assert len(new_predictions) == 0

    # inplace

    new_doc = move_annotations_from_predictions(
        document,
        layer_names=["labeled_spans"],
        inplace=True,
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 2

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 0


def test_move_annotations_to_predictions(document):
    new_doc = move_annotations_to_predictions(
        document,
        layer_names=["labeled_spans"],
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 2

    new_entities = new_doc["labeled_spans"]
    assert len(new_entities) == 0

    new_predictions = new_doc["labeled_spans"].predictions
    assert len(new_predictions) == 3

    # inplace

    new_doc = move_annotations_from_predictions(
        document,
        layer_names=["labeled_spans"],
        inplace=True,
    )

    original_entities = document["labeled_spans"]
    assert len(original_entities) == 2

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 0


def document_processor(document) -> TextBasedDocument:
    doc = document.copy()
    doc["labeled_spans"].append(LabeledSpan(start=0, end=0, label="empty"))
    return doc


def none_processor(document) -> None:
    return None


def test_process_documents(document):
    result = process_documents(
        documents=[document],
        processor=document_processor,
    )
    doc = result[0]

    spans = doc["labeled_spans"]
    assert len(spans) == 4

    result = process_documents(
        documents=[document],
        processor=none_processor,
    )
    doc = result[0]

    spans = doc["labeled_spans"]
    assert len(spans) == 3


def documents_processor(documents) -> TextBasedDocument:
    for doc in documents:
        doc["labeled_spans"].append(LabeledSpan(start=0, end=0, label="empty"))
    return documents


def test_process_pipeline_steps(document):
    docs = process_pipeline_steps(
        documents=[document], processors={"add_span": documents_processor}
    )

    doc = docs[0]

    spans = doc["labeled_spans"]
    original_spans = document["labeled_spans"]
    assert len(spans) == 4
    assert len(original_spans) == 3

    docs = process_pipeline_steps(
        documents=[document],
        processors={"add_span": documents_processor},
        inplace=True,
    )

    original_spans = document["labeled_spans"]
    assert len(original_spans) == 4


def test_add_annotations_from_other_documents(document, document_with_relations):

    original_relations = document_with_relations["binary_relations"]
    assert len(original_relations) == 2
    original_relations_predictions = document_with_relations["binary_relations"].predictions
    assert len(original_relations_predictions) == 1

    prepared_docs = add_annotations_from_other_documents(
        docs=[document], other_docs=[document_with_relations], layer_names=["binary_relations"]
    )

    prepared_doc = prepared_docs[0]

    relations = prepared_doc["binary_relations"]
    assert len(relations) == 2

    # from predictions

    prepared_docs = add_annotations_from_other_documents(
        docs=[document],
        other_docs=[document_with_relations],
        layer_names=["binary_relations"],
        from_predictions=True,
    )

    prepared_doc = prepared_docs[0]
    relations = prepared_doc["binary_relations"]
    assert len(relations) == 1

    # to predictions

    prepared_docs = add_annotations_from_other_documents(
        docs=[document],
        other_docs=[document_with_relations],
        layer_names=["binary_relations"],
        to_predictions=True,
    )

    prepared_doc = prepared_docs[0]
    relations = prepared_doc["binary_relations"].predictions
    assert len(relations) == 3  # adds to previously existing predictions

    # inplace

    prepared_docs = add_annotations_from_other_documents(
        docs=[document],
        other_docs=[document_with_relations],
        layer_names=["binary_relations"],
        inplace=True,
    )

    relations = document["binary_relations"]
    assert len(relations) == 2


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def old_document():
    document = TextDocumentWithEntitiesAndRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )

    document.entities.extend(
        [
            LabeledSpan(start=0, end=5, label="PERSON"),
            LabeledSpan(start=15, end=30, label="LOCATION"),
            LabeledSpan(start=44, end=48, label="ORGANIZATION"),
        ]
    )
    assert str(document.entities[0]) == "Harry"
    assert str(document.entities[1]) == "Berlin, Germany"
    assert str(document.entities[2]) == "DFKI"

    document.relations.extend(
        [
            BinaryRelation(
                head=document.entities[0],
                tail=document.entities[1],
                label="lives_in",
            ),
            BinaryRelation(
                head=document.entities[0],
                tail=document.entities[2],
                label="works_at",
            ),
        ]
    )

    return document


@pytest.mark.slow
def test_ner_re_pipeline(old_document):
    pipeline = NerRePipeline(
        ner_model_path="pie/example-ner-spanclf-conll03",
        re_model_path="pie/example-re-textclf-tacred",
        entity_layer="entities",
        relation_layer="relations",
        device=-1,
        batch_size=1,
        show_progress_bar=False,
    )

    docs = pipeline(documents=[old_document])
    assert len(docs) == 1

    doc = docs[0]
    predicted_entities = doc.entities.predictions
    assert len(predicted_entities) == 4
    predicted_relations = doc.relations.predictions
    assert len(predicted_relations) == 0
