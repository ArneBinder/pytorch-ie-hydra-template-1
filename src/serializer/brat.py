import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

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


def serialize_labeled_span(
    span_id: str,
    labeled_span: Union[LabeledMultiSpan, LabeledSpan],
    label_prefix: Optional[str] = None,
) -> str:
    """Serialize a labeled span into a string representation.

    Args:
        span_id (str): The identifier for the labeled span.
        labeled_span (Union[LabeledMultiSpan, LabeledSpan]): The labeled span object to serialize.
            It can be either a LabeledMultiSpan or a LabeledSpan.
        label_prefix (Optional[str], optional): A prefix to be added to the label.
            Defaults to None.

    Returns:
        str: The serialized representation of the labeled_span.

    Raises:
        Warning: If the labeled_span has an unknown type.
    """
    label = labeled_span.label if label_prefix is None else f"{label_prefix}-{labeled_span.label}"
    if isinstance(labeled_span, LabeledMultiSpan):
        locations = []
        texts = []
        for slice in labeled_span.slices:
            start_idx = slice[0]
            end_idx = slice[1]
            texts.append(labeled_span.target[start_idx:end_idx])
            locations.append(f"{start_idx} {end_idx}")
        location = ";".join(locations)
        text = " ".join(texts)
        serialized_labeled_span = f"{span_id}\t{label} {location}\t{text}\n"
    elif isinstance(labeled_span, LabeledSpan):
        start_idx = labeled_span.start
        end_idx = labeled_span.end
        entity_text = labeled_span.target[start_idx:end_idx]
        serialized_labeled_span = f"{span_id}\t{label} {start_idx} {end_idx}\t{entity_text}\n"
    else:
        raise Warning(f"labeled span has unknown type: {type(labeled_span)}")
    return serialized_labeled_span


def serialize_labeled_spans(
    labeled_spans: BaseAnnotationList,
    label_prefix: Optional[str] = None,
    first_span_id: int = 0,
) -> Tuple[List[str], Dict[LabeledSpan, str]]:
    """Converts entity annotations of type LabeledMultiSpan and LabeledSpan to annotations in the
    Brat format.

    Parameters:
        labeled_spans (BaseAnnotationList): The list of entity annotations.
        label_prefix (Optional[str]): An optional prefix to add to entity labels.
        first_span_id: An integer value used for creating span annotation IDs. It ensures the proper assignment of IDs
            for predicted annotations, particularly when gold annotations have already been included.

    Returns:
        Tuple[List[str], DefaultDict[LabeledSpan, str]]: A tuple containing a list of strings representing
        entity annotations in the Brat format, and a dictionary mapping labeled spans to their IDs.
    """
    span2id: Dict[LabeledSpan, str] = defaultdict()
    serialized_labeled_spans = []
    for i, labeled_span in enumerate(labeled_spans, start=first_span_id):
        span_id = f"T{i}"
        serialized_labeled_span = serialize_labeled_span(
            span_id=span_id, labeled_span=labeled_span, label_prefix=label_prefix
        )
        span2id[labeled_span] = span_id
        serialized_labeled_spans.append(serialized_labeled_span)
    return serialized_labeled_spans, span2id


def serialize_binary_relation(
    relation_id: str,
    binary_relation: Union[LabeledMultiSpan, LabeledSpan],
    span2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
) -> str:
    """Serialize a binary relation into a string representation.

    Args:
        relation_id (str): The identifier for the binary relation.
        binary_relation (Union[LabeledMultiSpan, LabeledSpan]): The binary relation object to serialize.
            Labeled Spans in the binary relation can be either a LabeledMultiSpan or a LabeledSpan.
        span2id (Dict[Annotation, str]): A dictionary mapping span annotations to their IDs.
        label_prefix (Optional[str], optional): A prefix to be added to the label.
            Defaults to None.

    Returns:
        str: The serialized representation of the binary relation.

    Raises:
        Warning: If the binary relation has an unknown type.
    """
    if not isinstance(binary_relation, BinaryRelation):
        raise Warning(f"relation has unknown type: {type(binary_relation)}")

    arg1 = span2id[binary_relation.head]
    arg2 = span2id[binary_relation.tail]
    label = (
        binary_relation.label
        if label_prefix is None
        else f"{label_prefix}-{binary_relation.label}"
    )
    serialized_binary_relation = f"{relation_id}\t{label} Arg1:{arg1} Arg2:{arg2}\n"
    return serialized_binary_relation


def serialize_binary_relations(
    binary_relations: BaseAnnotationList,
    span2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
    first_relation_id: int = 0,
) -> List[str]:
    """
    Converts relation annotations to annotations in the Brat format.
    e.g: R0 Arg1 Arg2 LABEL

    Parameters:
        binary_relations (BaseAnnotationList): The list of relation annotations.
        span2id (Dict[Annotation, str]): A dictionary mapping labeled spans to their annotation IDs.
        label_prefix (Optional[str]): An optional prefix to add to relation labels.
        first_relation_id: An integer value used for creating relation annotation IDs. It ensures the proper assignment
            of IDs for predicted annotations, particularly when gold annotations have already been included.

    Returns:
        List[str]: A list of strings representing relation annotations in the Brat format.
    """
    serialized_binary_relations = []
    for i, binary_relation in enumerate(binary_relations, start=first_relation_id):
        relation_id = f"R{i}"
        serialized_binary_relation = serialize_binary_relation(
            relation_id=relation_id,
            binary_relation=binary_relation,
            span2id=span2id,
            label_prefix=label_prefix,
        )
        serialized_binary_relations.append(serialized_binary_relation)

    return serialized_binary_relations


def serialize_annotations(
    labeled_span_layer: AnnotationLayer,
    binary_relation_layer: AnnotationLayer,
    gold_label_prefix: Optional[str] = None,
    prediction_label_prefix: Optional[str] = None,
) -> List[str]:
    """Serialize annotations including labeled spans and binary relations into a list of strings.

    Args:
        labeled_span_layer (AnnotationLayer): Annotation layer containing labeled spans.
        binary_relation_layer (AnnotationLayer): Annotation layer containing binary relations.
        gold_label_prefix (Optional[str], optional): Prefix to be added to gold labels.
            Defaults to None.
        prediction_label_prefix (Optional[str], optional): Prefix to be added to prediction labels.
            Defaults to None.

    Returns:
        List[str]: List of serialized annotations.
    """
    serialized_labeled_spans = []
    serialized_binary_relations = []
    if gold_label_prefix is not None:
        serialized_labeled_spans_gold, span2id = serialize_labeled_spans(
            labeled_spans=labeled_span_layer,
            label_prefix=gold_label_prefix,
        )
        serialized_labeled_spans.extend(serialized_labeled_spans_gold)
        serialized_binary_relations_gold = serialize_binary_relations(
            binary_relations=binary_relation_layer,
            span2id=span2id,
            label_prefix=gold_label_prefix,
        )
        serialized_binary_relations.extend(serialized_binary_relations_gold)
    else:
        span2id = {}
    serialized_labeled_spans_pred, span2id_pred = serialize_labeled_spans(
        labeled_spans=labeled_span_layer.predictions,
        label_prefix=prediction_label_prefix,
        first_span_id=len(serialized_labeled_spans),
    )
    span2id.update(span2id_pred)
    serialized_labeled_spans.extend(serialized_labeled_spans_pred)
    serialized_binary_relations_pred = serialize_binary_relations(
        binary_relations=binary_relation_layer.predictions,
        span2id=span2id,
        label_prefix=prediction_label_prefix,
        first_relation_id=len(serialized_binary_relations),
    )
    serialized_binary_relations.extend(serialized_binary_relations_pred)
    return serialized_labeled_spans + serialized_binary_relations


class BratSerializer(DocumentSerializer):
    """BratSerializer serialize documents into the Brat format. It requires "entity_layer" and
    "relation_layer" parameters which defines the entity and relation annotation layer names. If
    document processor is provided then these parameters must align with the respective entity and
    relation annotation layer of resulting document from the document processor. BratSerializer
    additionally provides the functionality to include both gold and predicted annotations in the
    resulting annotation file, with the option to differentiate them using the label_prefix.

    Attributes:
        entity_layer: The name of the entity annotation layer.
        relation_layer: The name of the relation annotation layer.
        document_processor: A function or callable object to process documents before serialization.
        prediction_label_prefix: An optional prefix for labels in predicted annotations.
        gold_label_prefix: An optional prefix for labels in gold annotations.
        default_kwargs: Additional keyword arguments to be used as defaults during serialization.
    """

    def __init__(
        self,
        entity_layer,
        relation_layer,
        document_processor=None,
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
        if self.document_processor is not None:
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
