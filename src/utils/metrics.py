import logging
from itertools import chain
from typing import Dict, List, Optional, Type, Union

from pytorch_ie.core import Document

from src.document.types import DocumentWithEntitiesRelationsAndLabeledPartitions
from src.metrics import F1Metric
from src.serializer import JsonSerializer

logger = logging.getLogger(__name__)


def evaluate_document_layer(
    path_or_documents: Union[str, List[Document]],
    layer: str,
    document_type: Optional[Type[Document]] = DocumentWithEntitiesRelationsAndLabeledPartitions,
    label_field: Optional[str] = "label",
    exclude_labels: Optional[List[str]] = None,
    exclude_annotation_fields: Optional[List[str]] = None,
    show_as_markdown: bool = True,
) -> Dict[str, Dict[str, float]]:
    if isinstance(path_or_documents, str):
        logger.warning(f"load documents from: {path_or_documents}")
        if document_type is None:
            raise Exception("document_type is required to load serialized documents")
        documents = JsonSerializer.read(file_name=path_or_documents, document_type=document_type)
    else:
        documents = path_or_documents
    if label_field is not None:
        labels = set(
            chain(*[[getattr(ann, label_field) for ann in doc[layer]] for doc in documents])
        )
        if exclude_labels is not None:
            labels = labels - set(exclude_labels)
    else:
        labels = None
    f1metric = F1Metric(
        layer=layer,
        label_field=label_field,
        labels=labels,
        exclude_annotation_fields=exclude_annotation_fields,
        show_as_markdown=show_as_markdown,
    )
    f1metric(documents)

    metric_values = f1metric.values()
    return metric_values
