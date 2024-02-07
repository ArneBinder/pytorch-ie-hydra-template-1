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
    label_prefix: Optional[str] = None,
    last_span_id: int = 0,
) -> Tuple[List[str], DefaultDict[LabeledSpan, str]]:
    """Converts entity annotations of type LabeledMultiSpan and LabeledSpan to annotations in the
    Brat format.

    Parameters:
        document (Document): The document containing the annotations.
        entities (Union[BaseAnnotationList, AnnotationLayer]): The list of entity annotations.
        label_prefix (Optional[str]): An optional prefix to add to entity labels.
        last_span_id: An integer value used for creating span annotation IDs. It ensures the proper assignment of IDs
            for predicted annotations, particularly when gold annotations have already been included.

    Returns:
        Tuple[List[str], DefaultDict[LabeledSpan, str]]: A tuple containing a list of strings representing
        entity annotations in the Brat format, and a dictionary mapping labeled spans to their IDs.
    """
    span2id: DefaultDict[LabeledSpan, str] = defaultdict()
    ann_entities = []
    for i, entity in enumerate(entities, start=last_span_id):
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
    last_rel_id: int = 0,
) -> List[str]:
    """
    Converts relation annotations to annotations in the Brat format.
    e.g: R0 Arg1 Arg2 LABEL

    Parameters:
        relations (Union[BaseAnnotationList, AnnotationLayer]): The list of relation annotations.
        span2id (DefaultDict[LabeledSpan, str]): A dictionary mapping labeled spans to their annotation IDs.
        label_prefix (Optional[str]): An optional prefix to add to relation labels.
        last_rel_id: An integer value used for creating relation annotation IDs. It ensures the proper assignment
            of IDs for predicted annotations, particularly when gold annotations have already been included.

    Returns:
        List[str]: A list of strings representing relation annotations in the Brat format.
    """
    ann_relations = []
    for i, relation in enumerate(relations, start=last_rel_id):
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
    file_descriptor,
    label_prefix: Optional[str] = None,
    last_span_id: int = 0,
    last_rel_id: int = 0,
) -> Tuple[List[str], List[str]]:
    """Writes annotations to a file descriptor in the Brat format.

    Parameters:
        document (Document): The document containing the annotations.
        entities (Union[BaseAnnotationList, AnnotationLayer]): The list of entity annotations.
        relations (Union[BaseAnnotationList, AnnotationLayer]): The list of relation annotations.
        label_prefix (Optional[str]): An optional prefix to add to relation labels.
        file_descriptor: The file descriptor to write the annotations to.
        last_span_id: An integer value used for creating span annotation IDs.
        last_rel_id: An integer value used for creating relation annotation IDs.

    Returns:
        Tuple[List[str],List[str]]: A tuple containing a list of strings representing entity annotations
        and a list of strings representing relation annotations in the Brat format.
    """
    ann_entities, span2id = spans_to_ann_entities(
        document=document,
        entities=entities,
        label_prefix=label_prefix,
        last_span_id=last_span_id,
    )
    ann_relations = relations_to_ann_relations(
        relations=relations,
        span2id=span2id,
        label_prefix=label_prefix,
        last_rel_id=last_rel_id,
    )
    for ann_entity in ann_entities:
        file_descriptor.write(ann_entity)
    for ann_relation in ann_relations:
        file_descriptor.write(ann_relation)

    return ann_entities, ann_relations


class BratSerializer(DocumentSerializer):
    """BratSerializer serialize documents into the Brat format. It requires a document processor
    which converts predicted annotations into the desired format. "entity_layer" and
    "relation_layer" parameters must align with the respective entity and relation annotation layer
    of resulting document from the document processor. BratSerializer additionally provides the
    functionality to include both gold and predicted annotations in the resulting annotation file,
    with the option to differentiate them using the label_prefix.

    Attributes:
        document_processor: A function or callable object to process documents before serialization.
        entity_layer: The name of the entity annotation layer.
        relation_layer: The name of the relation annotation layer.
        prediction_label_prefix: An optional prefix for labels in predicted annotations.
        gold_label_prefix: An optional prefix for labels in gold annotations.
        default_kwargs: Additional keyword arguments to be used as defaults during serialization.
    """

    def __init__(
        self,
        document_processor,
        entity_layer,
        relation_layer,
        prediction_label_prefix=None,
        gold_label_prefix=None,
        **kwargs,
    ):
        self.document_processor = document_processor
        self.entity_layer = entity_layer
        self.relation_layer = relation_layer
        self.prediction_label_prefix = prediction_label_prefix
        self.gold_label_prefix = gold_label_prefix
        self.default_kwargs = kwargs

    def __call__(self, documents: Sequence[Document], **kwargs) -> Dict[str, str]:
        documents = list(map(self.document_processor, documents))
        return self.write_with_defaults(
            documents=documents,
            entity_layer=self.entity_layer,
            relation_layer=self.relation_layer,
            prediction_label_prefix=self.prediction_label_prefix,
            gold_label_prefix=self.gold_label_prefix,
            **kwargs,
        )

    def write_with_defaults(self, **kwargs) -> Dict[str, str]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.write(**all_kwargs)

    @classmethod
    def write(
        cls,
        documents: Sequence[Document],
        entity_layer: str,
        relation_layer: str,
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
                last_span_id = 0
                last_rel_id = 0
                if gold_label_prefix is not None:
                    entities = doc[entity_layer]
                    relations = doc[relation_layer]
                    ann_entities, ann_relations = write_annotations_to_file(
                        document=doc,
                        entities=entities,
                        relations=relations,
                        file_descriptor=f,
                        label_prefix=gold_label_prefix,
                        last_span_id=last_span_id,
                        last_rel_id=last_rel_id,
                    )
                    last_span_id = len(ann_entities)
                    last_rel_id = len(ann_relations)

                entities = doc[entity_layer].predictions
                relations = doc[relation_layer].predictions
                write_annotations_to_file(
                    document=doc,
                    entities=entities,
                    relations=relations,
                    file_descriptor=f,
                    label_prefix=prediction_label_prefix,
                    last_span_id=last_span_id,
                    last_rel_id=last_rel_id,
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
