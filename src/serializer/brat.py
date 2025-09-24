import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from pie_core.utils.hydra import serialize_document_type
from pie_datasets.core.dataset_dict import METADATA_FILE_NAME
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationLayer, Document
from pytorch_ie.documents import TextBasedDocument

from src.serializer.interface import DocumentSerializer
from src.utils.logging_utils import get_pylogger

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
    annotation_ids: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[Annotation, str]]:
    serialized_annotations = []
    new_annotation2id: Dict[Annotation, str] = {}
    for idx, annotation in enumerate(annotations):
        annotation_type, serialized_annotation = serialize_annotation(
            annotation=annotation,
            annotation2id=annotation2id,
            label_prefix=label_prefix,
        )
        if annotation_ids is not None:
            if indices.get(annotation_type, 0) > 0:
                raise ValueError(
                    "Cannot specify annotation IDs for the same type (e.g. T or R) if there are "
                    "other annotations of the same type without an ID."
                )
            annotation_id = annotation_ids[idx]
        else:
            index = indices[annotation_type]
            annotation_id = f"{annotation_type}{index}"
            indices[annotation_type] += 1
        serialized_annotations.append(f"{annotation_id}\t{serialized_annotation}")
        new_annotation2id[annotation] = annotation_id

    return serialized_annotations, new_annotation2id


def serialize_annotation_layers(
    layers: List[Tuple[AnnotationLayer, str]],
    gold_label_prefix: Optional[str] = None,
    prediction_label_prefix: Optional[str] = None,
    gold_annotation_ids: Optional[List[Optional[List[str]]]] = None,
    prediction_annotation_ids: Optional[List[Optional[List[str]]]] = None,
) -> List[str]:
    """Serialize annotations from given annotation layers into a list of strings.

    Args:
        layers (List[Tuple[AnnotationLayer, str]]): Annotation layers to be serialized and what
            should be serialized, i.e. "gold", "prediction", or "both".
        gold_label_prefix (Optional[str], optional): Prefix to be added to gold labels.
            Defaults to None.
        prediction_label_prefix (Optional[str], optional): Prefix to be added to prediction labels.
            Defaults to None.
        gold_annotation_ids (Optional[List[Optional[str]]], optional): List of gold annotation IDs.
            If provided, the length should match the number of layers. Defaults to None.
        prediction_annotation_ids (Optional[List[Optional[str]]], optional): List of prediction
            annotation IDs. If provided, the length should match the number of layers. Defaults to None.

    Returns:
        List[str]: List of serialized annotations.
    """

    all_serialized_annotations = []
    gold_annotation2id: Dict[Annotation, str] = {}
    prediction_annotation2id: Dict[Annotation, str] = {}
    indices: Dict[str, int] = defaultdict(int)
    for idx, (layer, what) in enumerate(layers):
        if what not in ["gold", "prediction", "both"]:
            raise ValueError(
                f'Invalid value for what to serialize: "{what}". Expected "gold", "prediction", or "both".'
            )
        if (
            what == "both"
            and gold_label_prefix is None
            and prediction_label_prefix is None
            and len(layer) > 0
            and len(layer.predictions) > 0
        ):
            raise ValueError(
                "Cannot serialize both gold and prediction annotations without a label prefix for "
                "at least one of them. Consider setting gold_label_prefix or prediction_label_prefix."
            )
        serialized_annotations = []
        if what in ["gold", "both"]:
            if gold_annotation_ids is not None:
                if len(gold_annotation_ids) <= idx:
                    raise ValueError(
                        "gold_annotation_ids should have the same length as the number of layers."
                    )
                current_gold_annotation_ids = gold_annotation_ids[idx]
                if current_gold_annotation_ids is not None and len(
                    current_gold_annotation_ids
                ) != len(layer):
                    raise ValueError(
                        "gold_annotation_ids should have the same length as the number of gold annotations."
                    )
            else:
                current_gold_annotation_ids = None

            serialized_gold_annotations, new_gold_ann2id = serialize_annotations(
                annotations=layer,
                indices=indices,
                # gold annotations can only reference other gold annotations
                annotation2id=gold_annotation2id,
                label_prefix=gold_label_prefix,
                annotation_ids=current_gold_annotation_ids,
            )
            serialized_annotations.extend(serialized_gold_annotations)
            gold_annotation2id.update(new_gold_ann2id)
        if what in ["prediction", "both"]:
            if prediction_annotation_ids is not None:
                if len(prediction_annotation_ids) <= idx:
                    raise ValueError(
                        "prediction_annotation_ids should have the same length as the number of layers."
                    )
                current_prediction_annotation_ids = prediction_annotation_ids[idx]
                if current_prediction_annotation_ids is not None and len(
                    current_prediction_annotation_ids
                ) != len(layer.predictions):
                    raise ValueError(
                        "prediction_annotation_ids should have the same length as the number of prediction annotations."
                    )
            else:
                current_prediction_annotation_ids = None
            serialized_predicted_annotations, new_pred_ann2id = serialize_annotations(
                annotations=layer.predictions,
                indices=indices,
                # Predicted annotations can reference both gold and predicted annotations.
                # Note that predictions take precedence over gold annotations.
                annotation2id={**gold_annotation2id, **prediction_annotation2id},
                label_prefix=prediction_label_prefix,
                annotation_ids=current_prediction_annotation_ids,
            )
            prediction_annotation2id.update(new_pred_ann2id)
            serialized_annotations.extend(serialized_predicted_annotations)
        all_serialized_annotations.extend(serialized_annotations)
    return all_serialized_annotations


class BratSerializer(DocumentSerializer):
    """BratSerializer serialize documents into the Brat format. It requires a "layers" parameter to
    specify the annotation layers to serialize. For now, it supports layers containing LabeledSpan,
    LabeledMultiSpan, and BinaryRelation annotations.

    Attributes:
        layers: A mapping from annotation layer names that should be serialized to what should be
            serialized, i.e. "gold", "prediction", or "both".
        document_processor: A function or callable object to process documents before serialization.
        gold_label_prefix: If provided, gold annotations are serialized and its labels are prefixed
            with the given string. Otherwise, only predicted annotations are serialized.
        prediction_label_prefix: If provided, labels of predicted annotations are prefixed with the
            given string.
        metadata_gold_id_keys: A dictionary mapping layer names to metadata keys that contain the
            gold annotation IDs.
        metadata_prediction_id_keys: A dictionary mapping layer names to metadata keys that contain
            the prediction annotation IDs.
    """

    def __init__(
        self,
        layers: Dict[str, str],
        document_processor=None,
        **kwargs,
    ):
        self.document_processor = document_processor
        self.layers = layers
        self.default_kwargs = kwargs

    def __call__(
        self, documents: Iterable[Document], append: bool = False, **kwargs
    ) -> Dict[str, str]:
        if append:
            raise NotImplementedError("append mode is not yet implemented for BratSerializer")
        if self.document_processor is not None:
            documents = list(map(self.document_processor, documents))
        return self.write_with_defaults(
            documents=documents,
            layers=self.layers,
            **kwargs,
        )

    def write_with_defaults(self, **kwargs) -> Dict[str, str]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.write(**all_kwargs)

    @classmethod
    def write(
        cls,
        documents: Iterable[Document],
        layers: Dict[str, str],
        path: str,
        metadata_file_name: str = METADATA_FILE_NAME,
        split: Optional[str] = None,
        gold_label_prefix: Optional[str] = None,
        prediction_label_prefix: Optional[str] = None,
        metadata_gold_id_keys: Optional[Dict[str, str]] = None,
        metadata_prediction_id_keys: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:

        realpath = os.path.realpath(path)
        log.info(f'serialize documents to "{realpath}" ...')
        os.makedirs(realpath, exist_ok=True)

        if not isinstance(documents, Sequence):
            documents = list(documents)

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
            layer_names = list(layers)
            if metadata_gold_id_keys is not None:
                gold_annotation_ids = [
                    (
                        doc.metadata[metadata_gold_id_keys[layer_name]]
                        if layer_name in metadata_gold_id_keys
                        else None
                    )
                    for layer_name in layer_names
                ]
            else:
                gold_annotation_ids = None

            if metadata_prediction_id_keys is not None:
                prediction_annotation_ids = [
                    (
                        doc.metadata[metadata_prediction_id_keys[layer_name]]
                        if layer_name in metadata_prediction_id_keys
                        else None
                    )
                    for layer_name in layer_names
                ]
            else:
                prediction_annotation_ids = None
            serialized_annotations = serialize_annotation_layers(
                layers=[(doc[layer_name], layers[layer_name]) for layer_name in layer_names],
                gold_label_prefix=gold_label_prefix,
                prediction_label_prefix=prediction_label_prefix,
                gold_annotation_ids=gold_annotation_ids,
                prediction_annotation_ids=prediction_annotation_ids,
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
