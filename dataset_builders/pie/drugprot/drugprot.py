from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import datasets
from pie_datasets import GeneratorBasedBuilder
from pie_documents.annotations import BinaryRelation, LabeledSpan
from pie_documents.documents import (
    AnnotationLayer,
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    annotation_field,
)


@dataclass
class DrugprotDocument(TextBasedDocument):
    title: Optional[str] = None
    abstract: Optional[str] = None
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclass
class DrugprotBigbioDocument(TextBasedDocument):
    passages: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def example2drugprot(example: Dict[str, Any]) -> DrugprotDocument:
    metadata = {"entity_ids": [], "relation_ids": []}
    id2labeled_span: Dict[str, LabeledSpan] = {}

    document = DrugprotDocument(
        text=example["text"],
        title=example["title"],
        abstract=example["abstract"],
        id=example["document_id"],
        metadata=metadata,
    )

    for span in example["entities"]:
        labeled_span = LabeledSpan(
            start=span["offset"][0],
            end=span["offset"][1],
            label=span["type"],
        )
        document.entities.append(labeled_span)
        entity_id = span["id"].split("_")[1]
        document.metadata["entity_ids"].append(entity_id)
        id2labeled_span[entity_id] = labeled_span

    for relation in example["relations"]:
        arg1_id = relation["arg1_id"].split("_")[1]
        arg2_id = relation["arg2_id"].split("_")[1]
        document.relations.append(
            BinaryRelation(
                head=id2labeled_span[arg1_id],
                tail=id2labeled_span[arg2_id],
                label=relation["type"],
            )
        )
        relation_id = "R" + relation["id"].split("_")[1]
        document.metadata["relation_ids"].append(relation_id)

    return document


def example2drugprot_bigbio(example: Dict[str, Any]) -> DrugprotBigbioDocument:
    text = " ".join([" ".join(passage["text"]) for passage in example["passages"]])
    doc_id = example["document_id"]
    metadata = {"entity_ids": [], "relation_ids": []}
    id2labeled_span: Dict[str, LabeledSpan] = {}

    document = DrugprotBigbioDocument(
        text=text,
        id=doc_id,
        metadata=metadata,
    )
    for passage in example["passages"]:
        document.passages.append(
            LabeledSpan(
                start=passage["offsets"][0][0],
                end=passage["offsets"][0][1],
                label=passage["type"],
            )
        )
    # We sort labels and relation to always have a deterministic order for testing purposes.
    for span in example["entities"]:
        labeled_span = LabeledSpan(
            start=span["offsets"][0][0],
            end=span["offsets"][0][1],
            label=span["type"],
        )
        document.entities.append(labeled_span)
        entity_id = span["id"].split("_")[1]
        document.metadata["entity_ids"].append(entity_id)
        id2labeled_span[entity_id] = labeled_span

    for relation in example["relations"]:
        arg1_id = relation["arg1_id"].split("_")[1]
        arg2_id = relation["arg2_id"].split("_")[1]
        document.relations.append(
            BinaryRelation(
                head=id2labeled_span[arg1_id],
                tail=id2labeled_span[arg2_id],
                label=relation["type"],
            )
        )
        relation_id = "R" + relation["id"].split("_")[1]
        document.metadata["relation_ids"].append(relation_id)

    return document


def drugprot2example(doc: DrugprotDocument) -> Dict[str, Any]:
    entities = []
    for i, entity in enumerate(doc.entities):
        entities.append(
            {
                "id": doc.id + "_" + doc.metadata["entity_ids"][i],
                "type": entity.label,
                "text": doc.text[entity.start : entity.end],
                "offset": [entity.start, entity.end],
            }
        )

    relations = []
    for i, relation in enumerate(doc.relations):
        relations.append(
            {
                "id": doc.id + "_" + doc.metadata["relation_ids"][i][1:],
                "arg1_id": doc.id
                + "_"
                + doc.metadata["entity_ids"][doc.entities.index(relation.head)],
                "arg2_id": doc.id
                + "_"
                + doc.metadata["entity_ids"][doc.entities.index(relation.tail)],
                "type": relation.label,
            }
        )

    return {
        "document_id": doc.id,
        "title": doc.title,
        "abstract": doc.abstract,
        "text": doc.text,
        "entities": entities,
        "relations": relations,
    }


def drugprot_bigbio2example(doc: DrugprotBigbioDocument) -> Dict[str, Any]:
    entities = []
    for i, entity in enumerate(doc.entities):
        entities.append(
            {
                "id": doc.id + "_" + doc.metadata["entity_ids"][i],
                "normalized": [],
                "offsets": [[entity.start, entity.end]],
                "type": entity.label,
                "text": [doc.text[entity.start : entity.end]],
            }
        )

    relations = []
    for i, relation in enumerate(doc.relations):
        relations.append(
            {
                "id": doc.id + "_" + doc.metadata["relation_ids"][i][1:],
                "arg1_id": doc.id
                + "_"
                + doc.metadata["entity_ids"][doc.entities.index(relation.head)],
                "arg2_id": doc.id
                + "_"
                + doc.metadata["entity_ids"][doc.entities.index(relation.tail)],
                "normalized": [],
                "type": relation.label,
            }
        )

    passages = []
    for passage in doc.passages:
        passages.append(
            {
                "id": doc.id + "_" + passage.label,
                "text": [doc.text[passage.start : passage.end]],
                "offsets": [[passage.start, passage.end]],
                "type": passage.label,
            }
        )

    return {
        "coreferences": [],
        "document_id": doc.id,
        "entities": entities,
        "events": [],
        "id": doc.id,
        "passages": passages,
        "relations": relations,
    }


class Drugprot(GeneratorBasedBuilder):
    DOCUMENT_TYPES = {
        "drugprot_source": DrugprotDocument,
        "drugprot_bigbio_kb": DrugprotBigbioDocument,
    }

    BASE_DATASET_PATH = "bigbio/drugprot"
    # This revision includes the "test_background" split (see https://github.com/bigscience-workshop/biomedical/pull/928)
    BASE_DATASET_REVISION = "0cc98b3d292242e69adcfd2c3e5eea94baaca8ea"
    BASE_CONFIG_KWARGS_DICT = {
        config_name: {"trust_remote_code": True}
        for config_name in [None, "drugprot_source", "drugprot_bigbio_kb"]
    }

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="drugprot_source",
            version=datasets.Version("1.0.2"),
            description="DrugProt source version",
        ),
        datasets.BuilderConfig(
            name="drugprot_bigbio_kb",
            version=datasets.Version("1.0.0"),
            description="DrugProt BigBio version",
        ),
    ]

    @property
    def document_converters(self):
        if self.config.name == "drugprot_source":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: {
                    "entities": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        elif self.config.name == "drugprot_bigbio_kb":
            return {
                TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions: {
                    "passages": "labeled_partitions",
                    "entities": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        else:
            raise ValueError(f"Unknown dataset name: {self.config.name}")

    def _generate_document(
        self, example: Dict[str, Any], **kwargs
    ) -> Union[DrugprotDocument, DrugprotBigbioDocument]:
        if self.config.name == "drugprot_source":
            return example2drugprot(example)
        elif self.config.name == "drugprot_bigbio_kb":
            return example2drugprot_bigbio(example)
        else:
            raise ValueError(f"Unknown dataset config name: {self.config.name}")

    def _generate_example(
        self, document: Union[DrugprotDocument, DrugprotBigbioDocument], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(document, DrugprotBigbioDocument):
            return drugprot_bigbio2example(document)
        elif isinstance(document, DrugprotDocument):
            return drugprot2example(document)
        else:
            raise ValueError(f"Unknown document type: {type(document)}")
