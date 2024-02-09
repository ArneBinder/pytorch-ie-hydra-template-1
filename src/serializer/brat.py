import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from pie_datasets.core.dataset_dict import METADATA_FILE_NAME
from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.documents import TextBasedDocument
from pytorch_ie.core import Annotation, AnnotationLayer, Document
from pytorch_ie.utils.hydra import serialize_document_type

from src.serializer.interface import DocumentSerializer
from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


def serialize_labeled_span(
    annotation: LabeledSpan,
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Serialize a labeled span into a string representation.

    Args:
        annotation (LabeledSpan): The labeled span object to serialize.
        label_prefix (Optional[str], optional): A prefix to be added to the label. Defaults to None.

    Returns:
        str: The annotation type and serialized representation of the labeled span.
    """
    label = annotation.label if label_prefix is None else f"{label_prefix}-{annotation.label}"
    start_idx = annotation.start
    end_idx = annotation.end
    entity_text = annotation.target[start_idx:end_idx]
    serialized_labeled_span = f"{label} {start_idx} {end_idx}\t{entity_text}\n"
    return "T", serialized_labeled_span


def serialize_labeled_multi_span(
    annotation: LabeledMultiSpan,
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Serialize a labeled multi span into a string representation.

    Args:
        annotation (LabeledMultiSpan): The labeled multi span object to serialize.
        label_prefix (Optional[str], optional): A prefix to be added to the label. Defaults to None.

    Returns:
        str: The annotation type and serialized representation of the labeled multi span.
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
    return "T", serialized_labeled_span


def serialize_binary_relation(
    annotation: BinaryRelation,
    annotation2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Serialize a binary relation into a string representation.

    Args:
        annotation (Union[LabeledMultiSpan, LabeledSpan]): The binary relation object to serialize.
            Labeled Spans in the binary relation can be either a LabeledMultiSpan or a LabeledSpan.
        annotation2id (Dict[Annotation, str]): A dictionary mapping span annotations to their IDs.
        label_prefix (Optional[str], optional): A prefix to be added to the label.
            Defaults to None.

    Returns:
        str: The annotation type and serialized representation of the binary relation.
    """

    arg1 = annotation2id[annotation.head]
    arg2 = annotation2id[annotation.tail]
    label = annotation.label if label_prefix is None else f"{label_prefix}-{annotation.label}"
    serialized_binary_relation = f"{label} Arg1:{arg1} Arg2:{arg2}\n"
    return "R", serialized_binary_relation


def serialize_annotation(
    annotation: Annotation,
    annotation2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    if isinstance(annotation, LabeledMultiSpan):
        return serialize_labeled_multi_span(annotation=annotation, label_prefix=label_prefix)
    elif isinstance(annotation, LabeledSpan):
        return serialize_labeled_span(annotation=annotation, label_prefix=label_prefix)
    elif isinstance(annotation, BinaryRelation):
        return serialize_binary_relation(
            annotation=annotation, label_prefix=label_prefix, annotation2id=annotation2id
        )
    else:
        raise Warning(f"annotation has unknown type: {type(annotation)}")


def serialize_annotations(
    annotations: Iterable[Annotation],
    indices: Dict[str, int],
    annotation2id: Dict[Annotation, str],
    label_prefix: Optional[str] = None,
) -> Tuple[List[str], Dict[Annotation, str]]:
    serialized_annotations = []
    new_annotation2id: Dict[Annotation, str] = {}
    for annotation in annotations:
        annotation_type, serialized_annotation = serialize_annotation(
            annotation=annotation,
            annotation2id=annotation2id,
            label_prefix=label_prefix,
        )
        idx = indices[annotation_type]
        annotation_id = f"{annotation_type}{idx}"
        serialized_annotations.append(f"{annotation_id}\t{serialized_annotation}")
        new_annotation2id[annotation] = annotation_id
        indices[annotation_type] += 1

    return serialized_annotations, new_annotation2id


def serialize_annotation_layers(
    layers: List[AnnotationLayer],
    gold_label_prefix: Optional[str] = None,
    prediction_label_prefix: Optional[str] = None,
) -> List[str]:
    """Serialize annotations from given annotation layers into a list of strings.

    Args:
        layers (List[AnnotationLayer]): Annotation layers to be serialized.
        gold_label_prefix (Optional[str], optional): Prefix to be added to gold labels.
            Defaults to None.
        prediction_label_prefix (Optional[str], optional): Prefix to be added to prediction labels.
            Defaults to None.

    Returns:
        List[str]: List of serialized annotations.
    """
    all_serialized_annotations = []
    gold_annotation2id: Dict[Annotation, str] = {}
    prediction_annotation2id: Dict[Annotation, str] = {}
    indices: Dict[str, int] = defaultdict(int)
    for layer in layers:
        serialized_annotations = []
        if gold_label_prefix is not None:
            serialized_gold_annotations, new_gold_ann2id = serialize_annotations(
                annotations=layer,
                indices=indices,
                # gold annotations can only reference other gold annotations
                annotation2id=gold_annotation2id,
                label_prefix=gold_label_prefix,
            )
            serialized_annotations.extend(serialized_gold_annotations)
            gold_annotation2id.update(new_gold_ann2id)
        serialized_predicted_annotations, new_pred_ann2id = serialize_annotations(
            annotations=layer.predictions,
            indices=indices,
            # Predicted annotations can reference both gold and predicted annotations.
            # Note that predictions take precedence over gold annotations.
            annotation2id={**gold_annotation2id, **prediction_annotation2id},
            label_prefix=prediction_label_prefix,
        )
        prediction_annotation2id.update(new_pred_ann2id)
        serialized_annotations.extend(serialized_predicted_annotations)
        all_serialized_annotations.extend(serialized_annotations)
    return all_serialized_annotations


class BratSerializer(DocumentSerializer):
    """BratSerializer serialize documents into the Brat format. It requires a "layers" parameter to
    specify the annotation layers to serialize. For now, it supports layers containing LabeledSpan,
    LabeledMultiSpan, and BinaryRelation annotations.

    If a gold_label_prefix is provided, the gold annotations are serialized with the given prefix.
    Otherwise, only the predicted annotations are serialized. A document_processor can be provided
    to process documents before serialization.

    Attributes:
        layers: The names of the annotation layers to serialize.
        document_processor: A function or callable object to process documents before serialization.
        gold_label_prefix: If provided, gold annotations are serialized and its labels are prefixed
            with the given string. Otherwise, only predicted annotations are serialized.
        prediction_label_prefix: If provided, labels of predicted annotations are prefixed with the
            given string.
        default_kwargs: Additional keyword arguments to be used as defaults during serialization.
    """

    def __init__(
        self,
        layers: List[str],
        document_processor=None,
        prediction_label_prefix=None,
        gold_label_prefix=None,
        **kwargs,
    ):
        self.document_processor = document_processor
        self.layers = layers
        self.prediction_label_prefix = prediction_label_prefix
        self.gold_label_prefix = gold_label_prefix
        self.default_kwargs = kwargs

    def __call__(self, documents: Sequence[Document], **kwargs) -> Dict[str, str]:
        if self.document_processor is not None:
            documents = list(map(self.document_processor, documents))
        return self.write_with_defaults(
            documents=documents,
            layers=self.layers,
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
        layers: List[str],
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
            doc_id = getattr(doc, "id", None) or f"doc_{i}"
            if not isinstance(doc, TextBasedDocument):
                raise TypeError(
                    f"Document {doc_id} has unexpected type: {type(doc)}. "
                    "BratSerializer can only serialize TextBasedDocuments."
                )
            file_name = f"{doc_id}.ann"
            metadata_text[f"{file_name}"] = doc.text
            ann_path = os.path.join(realpath, file_name)
            serialized_annotations = serialize_annotation_layers(
                layers=[doc[layer] for layer in layers],
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
