import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

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
    idx: int,
    annotation: LabeledSpan,
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Serialize a labeled span into a string representation.

    Args:
        idx (str): The index for the labeled span.
        annotation (LabeledSpan): The labeled span object to serialize.
        label_prefix (Optional[str], optional): A prefix to be added to the label. Defaults to None.

    Returns:
        str: The id and serialized representation of the labeled span.
    """
    label = annotation.label if label_prefix is None else f"{label_prefix}-{annotation.label}"
    start_idx = annotation.start
    end_idx = annotation.end
    entity_text = annotation.target[start_idx:end_idx]
    serialized_labeled_span = f"{label} {start_idx} {end_idx}\t{entity_text}\n"
    return f"T{idx}", serialized_labeled_span


def serialize_labeled_multi_span(
    idx: int,
    annotation: LabeledMultiSpan,
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Serialize a labeled multi span into a string representation.

    Args:
        idx (int): The index for the labeled multi span.
        annotation (LabeledMultiSpan): The labeled multi span object to serialize.
        label_prefix (Optional[str], optional): A prefix to be added to the label. Defaults to None.

    Returns:
        str: The id and serialized representation of the labeled multi span.
    """
    label = annotation.label if label_prefix is None else f"{label_prefix}-{annotation.label}"

    locations = []
    texts = []
    for slice in annotation.slices:
        start_idx = slice[0]
        end_idx = slice[1]
        texts.append(annotation.target[start_idx:end_idx])
        locations.append(f"{start_idx} {end_idx}")
    location = ";".join(locations)
    text = " ".join(texts)
    serialized_labeled_span = f"{label} {location}\t{text}\n"
    return f"T{idx}", serialized_labeled_span


def serialize_binary_relation(
    idx: int,
    annotation: BinaryRelation,
    annotation2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Serialize a binary relation into a string representation.

    Args:
        idx (str): The index for the binary relation.
        annotation (Union[LabeledMultiSpan, LabeledSpan]): The binary relation object to serialize.
            Labeled Spans in the binary relation can be either a LabeledMultiSpan or a LabeledSpan.
        annotation2id (Dict[Annotation, str]): A dictionary mapping span annotations to their IDs.
        label_prefix (Optional[str], optional): A prefix to be added to the label.
            Defaults to None.

    Returns:
        str: The id and serialized representation of the binary relation.
    """
    if not isinstance(annotation, BinaryRelation):
        raise Warning(f"relation has unknown type: {type(annotation)}")

    arg1 = annotation2id[annotation.head]
    arg2 = annotation2id[annotation.tail]
    label = annotation.label if label_prefix is None else f"{label_prefix}-{annotation.label}"
    serialized_binary_relation = f"{label} Arg1:{arg1} Arg2:{arg2}\n"
    return f"R{idx}", serialized_binary_relation


def serialize_annotation(
    idx: int,
    annotation: Annotation,
    annotation2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    if isinstance(annotation, LabeledMultiSpan):
        return serialize_labeled_multi_span(
            idx=idx, annotation=annotation, label_prefix=label_prefix
        )
    elif isinstance(annotation, LabeledSpan):
        return serialize_labeled_span(idx=idx, annotation=annotation, label_prefix=label_prefix)
    elif isinstance(annotation, BinaryRelation):
        return serialize_binary_relation(
            idx=idx, annotation=annotation, label_prefix=label_prefix, annotation2id=annotation2id
        )
    else:
        raise Warning(f"annotation has unknown type: {type(annotation)}")


def serialize_annotations(
    annotations: Iterable[Annotation],
    annotation2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
    first_idx: int = 0,
) -> List[str]:
    serialized_annotations = []
    for i, annotation in enumerate(annotations, start=first_idx):
        annotation_id, serialized_annotation = serialize_annotation(
            idx=i,
            annotation=annotation,
            annotation2id=annotation2id,
            label_prefix=label_prefix,
        )
        serialized_annotations.append(f"{annotation_id}\t{serialized_annotation}")
        annotation2id[annotation] = annotation_id

    return serialized_annotations


def serialize_annotation_layers(
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
    annotation2id: Dict[Annotation, str] = {}
    if gold_label_prefix is not None:
        serialized_labeled_spans_gold = serialize_annotations(
            annotations=labeled_span_layer,
            label_prefix=gold_label_prefix,
            annotation2id=annotation2id,
        )
        serialized_labeled_spans.extend(serialized_labeled_spans_gold)
        serialized_binary_relations_gold = serialize_annotations(
            annotations=binary_relation_layer,
            annotation2id=annotation2id,
            label_prefix=gold_label_prefix,
        )
        serialized_binary_relations.extend(serialized_binary_relations_gold)
    else:
        annotation2id = {}
    serialized_labeled_spans_pred = serialize_annotations(
        annotations=labeled_span_layer.predictions,
        label_prefix=prediction_label_prefix,
        first_idx=len(serialized_labeled_spans),
        annotation2id=annotation2id,
    )
    serialized_labeled_spans.extend(serialized_labeled_spans_pred)
    serialized_binary_relations_pred = serialize_annotations(
        annotations=binary_relation_layer.predictions,
        annotation2id=annotation2id,
        label_prefix=prediction_label_prefix,
        first_idx=len(serialized_binary_relations),
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
            serialized_annotations = serialize_annotation_layers(
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
