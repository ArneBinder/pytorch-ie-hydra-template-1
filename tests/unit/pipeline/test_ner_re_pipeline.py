import dataclasses

import pytest
from pytorch_ie import AnnotationLayer
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import annotation_field
from pytorch_ie.documents import TextBasedDocument

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

    document = document.copy()

    document.binary_relations.predictions.extend(
        [
            BinaryRelation(
                head=document.labeled_spans[0],
                tail=document.labeled_spans[2],
                label="per:employee_of",
            ),
        ]
    )

    document.binary_relations.extend(
        [
            BinaryRelation(
                head=document.labeled_spans[0],
                tail=document.labeled_spans[2],
                label="per:employee_of",
            ),
        ]
    )

    return document


def test_clear_annotation_layers(document):
    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 2

    clear_annotation_layers(
        document,
        layer_names=["labeled_spans"],
    )

    new_entities = document["labeled_spans"]
    assert len(new_entities) == 0

    predictions = document["labeled_spans"].predictions
    assert len(predictions) == 2

    # clear predictions
    clear_annotation_layers(
        document,
        layer_names=["labeled_spans"],
        predictions=True,
    )

    new_entities = document["labeled_spans"]
    assert len(new_entities) == 0

    new_predictions = document["labeled_spans"].predictions
    assert len(new_predictions) == 0


def test_move_annotations_from_predictions(document):
    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 2

    move_annotations_from_predictions(
        document,
        layer_names=["labeled_spans"],
    )

    new_entities = document["labeled_spans"]
    assert len(new_entities) == 2

    new_predictions = document["labeled_spans"].predictions
    assert len(new_predictions) == 0


def test_move_annotations_to_predictions(document):
    original_entities = document["labeled_spans"]
    assert len(original_entities) == 3

    original_predictions = document["labeled_spans"].predictions
    assert len(original_predictions) == 2

    move_annotations_to_predictions(
        document,
        layer_names=["labeled_spans"],
    )

    new_entities = document["labeled_spans"]
    assert len(new_entities) == 0

    new_predictions = document["labeled_spans"].predictions
    assert len(new_predictions) == 3


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
    original_spans = document["labeled_spans"]
    assert len(original_spans) == 3

    process_pipeline_steps(
        documents=[document],
        processors={"add_span": documents_processor},
    )

    original_spans = document["labeled_spans"]
    assert len(original_spans) == 4


def test_add_annotations_from_other_documents(document, document_with_relations):

    original_relations = document_with_relations["binary_relations"]
    assert len(original_relations) == 1
    original_relations_predictions = document_with_relations["binary_relations"].predictions
    assert len(original_relations_predictions) == 1

    add_annotations_from_other_documents(
        docs=[document], other_docs=[document_with_relations], layer_names=["binary_relations"]
    )

    relations = document["binary_relations"]
    assert len(relations) == 1

    # from predictions

    add_annotations_from_other_documents(
        docs=[document],
        other_docs=[document_with_relations],
        layer_names=["binary_relations"],
        from_predictions=True,
    )

    relations = document["binary_relations"]
    assert len(relations) == 1

    assert relations[0] == document_with_relations["binary_relations"].predictions[0]

    # to predictions

    add_annotations_from_other_documents(
        docs=[document],
        other_docs=[document_with_relations],
        layer_names=["binary_relations"],
        to_predictions=True,
    )

    relations = document["binary_relations"].predictions
    assert len(relations) == 1

    assert relations[0] == document_with_relations["binary_relations"].predictions[0]


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.mark.slow
@pytest.mark.parametrize("use_deprecated", [False, True])
def test_ner_re_pipeline(use_deprecated: bool, caplog):
    # These imports register the respective taskmodules and models for NER and RE
    from pytorch_ie.models import (
        TransformerSpanClassificationModel,
        TransformerTextClassificationModel,
    )
    from pytorch_ie.taskmodules import (
        TransformerRETextClassificationTaskModule,
        TransformerSpanClassificationTaskModule,
    )

    document = TextDocumentWithEntitiesAndRelations(
        text="Harry lives in Berlin, Germany. He works at DFKI.", id="tmp"
    )

    document.entities.extend(
        [
            LabeledSpan(start=0, end=5, label="PER"),
            LabeledSpan(start=15, end=30, label="LOC"),
            LabeledSpan(start=44, end=48, label="ORG"),
        ]
    )
    assert str(document.entities[0]) == "Harry"
    assert str(document.entities[1]) == "Berlin, Germany"
    assert str(document.entities[2]) == "DFKI"

    document.relations.extend(
        [
            BinaryRelation(
                head=document.entities[0],
                tail=document.entities[2],
                label="per:employee_of",
            ),
        ]
    )
    re_pipeline_kwargs = {
        "taskmodule_kwargs": {"create_relation_candidates": True},
        "pipeline_type": "pytorch-ie",
    }
    ner_pipeline_kwargs = {"pipeline_type": "pytorch-ie"}
    pipeline_kwargs = dict(
        ner_pipeline=ner_pipeline_kwargs,
        re_pipeline=re_pipeline_kwargs,
        entity_layer="entities",
        relation_layer="relations",
        device=-1,
        batch_size=1,
        show_progress_bar=False,
    )
    if use_deprecated:
        pipeline_kwargs["ner_model_path"] = "pie/example-ner-spanclf-conll03"
        pipeline_kwargs["re_model_path"] = "pie/example-re-textclf-tacred"
        # check warnings for deprecated parameters
        caplog.clear()
        pipeline = NerRePipeline(**pipeline_kwargs)
        assert (
            "Parameter ner_model_path is deprecated. Use individual processor arguments to provide "
            "the model path instead, i.e., use ner_pipeline.pretrained_model_name_or_path to set "
            "the NER model path." in caplog.messages
        )
        assert (
            "Parameter re_model_path is deprecated. Use individual processor arguments to provide "
            "the model path instead, i.e., use re_pipeline.pretrained_model_name_or_path to set "
            "the RE model path." in caplog.messages
        )
    else:
        pipeline_kwargs["ner_pipeline"][
            "pretrained_model_name_or_path"
        ] = "pie/example-ner-spanclf-conll03"
        pipeline_kwargs["re_pipeline"][
            "pretrained_model_name_or_path"
        ] = "pie/example-re-textclf-tacred"
        caplog.clear()
        pipeline = NerRePipeline(**pipeline_kwargs)
        assert (
            "Parameter ner_model_path is deprecated. Use individual processor arguments to provide the model path instead, i.e., use ner_pipeline.pretrained_model_name_or_path to set the NER model path."
            not in caplog.messages
        )
        assert (
            "Parameter re_model_path is deprecated. Use individual processor arguments to provide the model path instead, i.e., use re_pipeline.pretrained_model_name_or_path to set the RE model path."
            not in caplog.messages
        )

    docs = pipeline(documents=[document])
    assert len(docs) == 1

    doc: TextDocumentWithEntitiesAndRelations = docs[0]

    # gold entities and relations
    gold_entities = doc.entities
    assert len(gold_entities) == 3
    gold_relations = doc.relations
    assert len(gold_relations) == 1

    # predicted entities and relations
    predicted_entities = doc.entities.predictions
    assert len(predicted_entities) == 4

    assert str(predicted_entities[0]) == "Harry"
    assert predicted_entities[0].label == "PER"

    assert str(predicted_entities[1]) == "Berlin"
    assert predicted_entities[1].label == "LOC"

    assert str(predicted_entities[2]) == "Germany"
    assert predicted_entities[2].label == "LOC"

    assert str(predicted_entities[3]) == "DFKI"
    assert predicted_entities[3].label == "ORG"

    predicted_relations = doc.relations.predictions
    assert len(predicted_relations) == 6

    assert str(predicted_relations[0].head) == "Harry"
    assert str(predicted_relations[0].tail) == "Berlin"
    assert predicted_relations[0].label == "per:cities_of_residence"

    assert str(predicted_relations[1].head) == "Harry"
    assert str(predicted_relations[1].tail) == "Germany"
    assert predicted_relations[1].label == "per:countries_of_residence"

    assert str(predicted_relations[2].head) == "Harry"
    assert str(predicted_relations[2].tail) == "DFKI"
    assert predicted_relations[2].label == "per:employee_of"

    assert str(predicted_relations[3].head) == "Berlin"
    assert str(predicted_relations[3].tail) == "Harry"
    assert predicted_relations[3].label == "per:cities_of_residence"

    assert str(predicted_relations[4].head) == "Germany"
    assert str(predicted_relations[4].tail) == "Harry"
    assert predicted_relations[4].label == "per:countries_of_residence"

    assert str(predicted_relations[5].head) == "DFKI"
    assert str(predicted_relations[5].tail) == "Harry"
    assert predicted_relations[5].label == "per:employee_of"
