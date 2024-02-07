import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple, TypeVar

from pie_datasets.core.dataset_dict import METADATA_FILE_NAME
from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.documents import TextBasedDocument
from pytorch_ie import AnnotationLayer
from pytorch_ie.core import Document
from pytorch_ie.core.document import Annotation, BaseAnnotationList
from pytorch_ie.utils.hydra import serialize_document_type

from src.serializer.interface import DocumentSerializer
from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


def serialize_labeled_spans(
    entities: BaseAnnotationList,
    label_prefix: Optional[str] = None,
    first_span_id: int = 0,
) -> Tuple[List[str], Dict[LabeledSpan, str]]:
    """Converts entity annotations of type LabeledMultiSpan and LabeledSpan to annotations in the
    Brat format.

    Parameters:
        entities (BaseAnnotationList): The list of entity annotations.
        label_prefix (Optional[str]): An optional prefix to add to entity labels.
        first_span_id: An integer value used for creating span annotation IDs. It ensures the proper assignment of IDs
            for predicted annotations, particularly when gold annotations have already been included.

    Returns:
        Tuple[List[str], DefaultDict[LabeledSpan, str]]: A tuple containing a list of strings representing
        entity annotations in the Brat format, and a dictionary mapping labeled spans to their IDs.
    """
    span2id: Dict[LabeledSpan, str] = defaultdict()
    ann_entities = []
    for i, entity in enumerate(entities, start=first_span_id):
        entity_id = f"T{i}"
        label = entity.label if label_prefix is None else f"{label_prefix}-{entity.label}"
        if isinstance(entity, LabeledMultiSpan):
            locations = []
            texts = []
            for slice in entity.slices:
                start_idx = slice[0]
                end_idx = slice[1]
                texts.append(entity.target[start_idx:end_idx])
                locations.append(f"{start_idx} {end_idx}")
            location = ";".join(locations)
            text = " ".join(texts)
            entry = f"{entity_id}\t{label} {location}\t{text}\n"
        elif isinstance(entity, LabeledSpan):
            start_idx = entity.start
            end_idx = entity.end
            entity_text = entity.target[start_idx:end_idx]
            entry = f"{entity_id}\t{label} {start_idx} {end_idx}\t{entity_text}\n"
        else:
            raise Warning(f"labeled span has unknown type: {type(entity)}")
        span2id[entity] = entity_id
        ann_entities.append(entry)
    return ann_entities, span2id


def serialize_binary_relations(
    relations: BaseAnnotationList,
    span2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
    first_relation_id: int = 0,
) -> List[str]:
    """
    Converts relation annotations to annotations in the Brat format.
    e.g: R0 Arg1 Arg2 LABEL

    Parameters:
        relations (BaseAnnotationList): The list of relation annotations.
        span2id (Dict[Annotation, str]): A dictionary mapping labeled spans to their annotation IDs.
        label_prefix (Optional[str]): An optional prefix to add to relation labels.
        first_relation_id: An integer value used for creating relation annotation IDs. It ensures the proper assignment
            of IDs for predicted annotations, particularly when gold annotations have already been included.

    Returns:
        List[str]: A list of strings representing relation annotations in the Brat format.
    """
    ann_relations = []
    for i, relation in enumerate(relations, start=first_relation_id):
        if not isinstance(relation, BinaryRelation):
            raise Warning(f"relation has unknown type: {type(relation)}")
        relation_id = f"R{i}"
        arg1 = span2id[relation.head]
        arg2 = span2id[relation.tail]
        label = relation.label if label_prefix is None else f"{label_prefix}-{relation.label}"
        entry = f"{relation_id}\t{label} Arg1:{arg1} Arg2:{arg2}\n"
        ann_relations.append(entry)

    return ann_relations


def serialize_annotations(
    labeled_span_layer: AnnotationLayer,
    binary_relation_layer: AnnotationLayer,
    gold_label_prefix: Optional[str] = None,
    prediction_label_prefix: Optional[str] = None,
) -> List[str]:
    serialized_labeled_spans = []
    serialized_binary_relations = []
    if gold_label_prefix is not None:
        serialized_labeled_spans_gold, span2id = serialize_labeled_spans(
            entities=labeled_span_layer,
            label_prefix=gold_label_prefix,
            first_span_id=0,
        )
        serialized_labeled_spans.extend(serialized_labeled_spans_gold)
        serialized_binary_relations_gold = serialize_binary_relations(
            relations=binary_relation_layer,
            span2id=span2id,
            label_prefix=gold_label_prefix,
            first_relation_id=0,
        )
        serialized_binary_relations.extend(serialized_binary_relations_gold)
        last_span_id = len(serialized_labeled_spans_gold)
        last_rel_id = len(serialized_binary_relations_gold)
    else:
        last_span_id = 0
        last_rel_id = 0
        span2id = {}
    serialized_labeled_spans_pred, span2id_pred = serialize_labeled_spans(
        entities=labeled_span_layer.predictions,
        label_prefix=prediction_label_prefix,
        first_span_id=last_span_id,
    )
    span2id.update(span2id_pred)
    serialized_labeled_spans.extend(serialized_labeled_spans_pred)
    serialized_binary_relations_pred = serialize_binary_relations(
        relations=binary_relation_layer.predictions,
        span2id=span2id,
        label_prefix=prediction_label_prefix,
        first_relation_id=last_rel_id,
    )
    serialized_binary_relations.extend(serialized_binary_relations_pred)
    return serialized_labeled_spans + serialized_binary_relations


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
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{i}"
            if not isinstance(doc, TextBasedDocument):
                raise TypeError(
                    f"Document {doc_id} has unexpected type: {type(doc)}. "
                    "BratSerializer can only serialize TextBasedDocuments."
                )
            file_name = f"{doc_id}.ann"
            metadata_text[f"{file_name}"] = doc.text
            ann_path = os.path.join(realpath, file_name)
            serialized_annotations = serialize_annotations(
                labeled_span_layer=doc[entity_layer],
                binary_relation_layer=doc[relation_layer],
                gold_label_prefix=gold_label_prefix,
                prediction_label_prefix=prediction_label_prefix,
            )
            with open(ann_path, "w+") as f:
                f.writelines(serialized_annotations)

        metadata["text"] = metadata_text

        if os.path.exists(full_metadata_file_name):
            log.warning(
                f"metadata file {full_metadata_file_name} already exists, "
                "it will be overwritten!"
            )
        with open(full_metadata_file_name, "w") as f:
            json.dump(metadata, f, indent=2)
        return {"path": realpath, "metadata_file_name": metadata_file_name}
