import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import logging
from itertools import chain
from typing import Callable, Dict, List, Optional, Type, Union

import pandas as pd
from pytorch_ie.core import Document
from pytorch_ie.metrics import F1Metric
from pytorch_ie.utils.hydra import resolve_target

from src.document.types import DocumentWithEntitiesRelationsAndLabeledPartitions
from src.serializer import JsonSerializer

logger = logging.getLogger(__name__)


def evaluate_document_layer(
    path_or_documents: Union[str, List[Document]],
    layer: str,
    document_type: Optional[Type[Document]] = DocumentWithEntitiesRelationsAndLabeledPartitions,
    label_field: Optional[str] = "label",
    exclude_labels: Optional[List[str]] = None,
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
        show_as_markdown=show_as_markdown,
    )
    f1metric(documents)

    metric_values = f1metric.compute()
    return metric_values


def get_document_converter(document_converter: str) -> Callable:
    raise NotImplementedError(f"unknown document converter: {document_converter}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serialized_documents",
        type=str,
        nargs="+",
        required=True,
        help="file name of serialized documents in jsonl format",
    )
    parser.add_argument("--layer", type=str, required=True, help="annotation layer to evaluate")
    parser.add_argument(
        "--label_field",
        type=str,
        default="label",
        help="Compute metrics per label. This requires the layer to contain annotations with that field.",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Do not compute metrics per label. Enable this flag if the layer does not contain annotations "
        "with a label field.",
    )
    parser.add_argument(
        "--document_type",
        type=resolve_target,
        default=DocumentWithEntitiesRelationsAndLabeledPartitions,
        help="document type to load serialized documents",
    )
    parser.add_argument(
        "--exclude_labels",
        type=str,
        nargs="+",
        default=["no_relation"],
        help="labels to exclude from evaluation",
    )
    parser.add_argument(
        "--preprocess_documents",
        type=get_document_converter,
        default=None,
        help="document converter function to preprocess documents",
    )

    args = parser.parse_args()

    # show info messages
    logging.basicConfig(level=logging.INFO)

    all_metric_values = []
    for file_name in args.serialized_documents:
        logger.info(f"evaluating {file_name} ...")
        documents = JsonSerializer.read(
            file_name=file_name,
            document_type=args.document_type,
        )
        if args.preprocess_documents is not None:
            documents = [args.preprocess_documents(document=document) for document in documents]

        metric_values = evaluate_document_layer(
            path_or_documents=documents,
            layer=args.layer,
            label_field=args.label_field if not args.no_labels else None,
            exclude_labels=args.exclude_labels,
        )
        all_metric_values.append(pd.DataFrame(metric_values).T)

    if len(all_metric_values) > 1:
        # mean and stddev over all metric results
        grouped_metric_values = pd.concat(all_metric_values).groupby(level=0)
        logger.info(f"aggregated results (n={len(all_metric_values)}):")
        logger.info(f"\nmean:\n{grouped_metric_values.mean().round(3).to_markdown()}")
        logger.info(f"\nstddev:\n{grouped_metric_values.std().round(3).to_markdown()}")
