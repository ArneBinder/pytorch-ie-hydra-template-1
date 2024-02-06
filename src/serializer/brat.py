import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from pie_datasets.core.dataset_dict import METADATA_FILE_NAME
from pie_modules.annotations import LabeledMultiSpan, LabeledSpan
from pytorch_ie import AnnotationLayer
from pytorch_ie.core import Document
from pytorch_ie.core.document import BaseAnnotationList
from pytorch_ie.utils.hydra import serialize_document_type

from src.serializer.interface import DocumentSerializer
from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


def spans_to_ann_entities(
    document: Document,
    entities: Union[BaseAnnotationList, AnnotationLayer],
    label_prefix=None,
) -> Tuple[List[str], DefaultDict[LabeledSpan, str]]:
    span2id: DefaultDict[LabeledSpan, str] = defaultdict()
    ann_entities = []
    for i, entity in enumerate(entities):
        entity_id = f"T{i}"
        label = entity.label if label_prefix is None else f"{label_prefix}-{entity.label}"
        if isinstance(entity, LabeledMultiSpan):
            locations = []
            texts = []
            for slice in entity.slices:
                start_idx = slice[0]
                end_idx = slice[1]
                texts.append(document.text[start_idx:end_idx])
                locations.append(f"{start_idx} {end_idx}")
            location = ";".join(locations)
            text = " ".join(texts)
            entry = f"{entity_id}\t{label} {location}\t{text}\n"
        elif isinstance(entity, LabeledSpan):
            start_idx = entity.start
            end_idx = entity.end
            entity_text = document.text[start_idx:end_idx]
            entry = f"{entity_id}\t{label} {start_idx} {end_idx}\t{entity_text}\n"
        else:
            raise Warning(f"Unknown entity type: {type(entity)}")
        span2id[entity] = entity_id
        ann_entities.append(entry)
    return ann_entities, span2id


def relations_to_ann_relations(
    relations: Union[BaseAnnotationList, AnnotationLayer],
    span2id: DefaultDict[LabeledSpan, str],
    label_prefix: Optional[str] = None,
) -> List[str]:
    ann_relations = []
    for i, relation in enumerate(relations):
        relation_id = f"R{i}"
        arg1 = span2id[relation.head]
        arg2 = span2id[relation.tail]
        label = relation.label if label_prefix is None else f"{label_prefix}-{relation.label}"
        entry = f"{relation_id}\t{label} Arg1:{arg1} Arg2:{arg2}\n"
        ann_relations.append(entry)

    return ann_relations


def write_annotations_to_file(
    document: Document,
    entities: Union[BaseAnnotationList, AnnotationLayer],
    relations: Union[BaseAnnotationList, AnnotationLayer],
    label_prefix: str,
    file_descriptor,
):
    ann_entities, span2id = spans_to_ann_entities(document, entities, label_prefix=label_prefix)
    ann_relations = relations_to_ann_relations(relations, span2id)
    for ann_entity in ann_entities:
        file_descriptor.write(ann_entity)
    for ann_relation in ann_relations:
        file_descriptor.write(ann_relation)


class BratSerializer(DocumentSerializer):
    def __init__(
        self, document_processor, prediction_label_prefix=None, gold_label_prefix=None, **kwargs
    ):
        self.document_processor = document_processor
        self.prediction_label_prefix = prediction_label_prefix
        self.gold_label_prefix = gold_label_prefix
        self.default_kwargs = kwargs

    def __call__(self, documents: Sequence[Document], **kwargs) -> Dict[str, str]:
        documents = list(map(self.document_processor, documents))
        if self.document_processor.result_document_type is not None:
            result_field_mapping = self.document_processor.result_field_mapping
        else:
            #  default as BratDocument
            result_field_mapping = {"labeled_spans": "spans", "binary_relations": "relations"}
        return self.write_with_defaults(
            documents=documents,
            prediction_label_prefix=self.prediction_label_prefix,
            gold_label_prefix=self.gold_label_prefix,
            result_field_mapping=result_field_mapping,
            **kwargs,
        )

    def write_with_defaults(self, **kwargs) -> Dict[str, str]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.write(**all_kwargs)

    @classmethod
    def write(
        cls,
        documents: Sequence[Document],
        result_field_mapping: Dict[str, str],
        path: str,
        metadata_file_name: str = METADATA_FILE_NAME,
        split: Optional[str] = None,
        gold_label_prefix: Optional[str] = None,
        prediction_label_prefix: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:

        realpath = os.path.realpath(path)
        log.info(f'serialize documents to "{realpath}" ...')
        os.makedirs(realpath, exist_ok=True)

        if len(documents) == 0:
            raise Exception("cannot serialize empty list of documents")
        document_type = type(documents[0])
        metadata = {"document_type": serialize_document_type(document_type)}
        full_metadata_file_name = os.path.join(realpath, metadata_file_name)

        if split is not None:
            realpath = os.path.join(realpath, split)
            os.makedirs(realpath, exist_ok=True)
        metadata_text = defaultdict(str)
        for doc in documents:
            file_name = f"{doc.id}.ann"
            metadata_text[f"{file_name}"] = doc.text
            ann_path = os.path.join(realpath, file_name)
            with open(ann_path, "w+") as f:
                if gold_label_prefix is not None:
                    entities = doc[result_field_mapping["labeled_spans"]]
                    relations = doc[result_field_mapping["binary_relations"]]
                    write_annotations_to_file(
                        document=doc,
                        entities=entities,
                        relations=relations,
                        file_descriptor=f,
                        label_prefix=gold_label_prefix,
                    )
                if prediction_label_prefix is not None:
                    entities = doc[result_field_mapping["labeled_spans"]].predictions
                    relations = doc[result_field_mapping["binary_relations"]].predictions
                    write_annotations_to_file(
                        document=doc,
                        entities=entities,
                        relations=relations,
                        file_descriptor=f,
                        label_prefix=prediction_label_prefix,
                    )
            f.close()

        metadata["text"] = metadata_text
        print(realpath)

        if os.path.exists(full_metadata_file_name):
            log.warning(
                f"metadata file {full_metadata_file_name} already exists, "
                "it will be overwritten!"
            )
        with open(full_metadata_file_name, "w") as f:
            json.dump(metadata, f, indent=2)
        return {"path": realpath, "metadata_file_name": metadata_file_name}
