import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import logging
from typing import Callable, Type, Union

import pandas as pd
from hydra._internal.instantiate._instantiate2 import _resolve_target

from src.document.types import DocumentWithEntitiesRelationsAndLabeledPartitions
from src.serializer import JsonSerializer
from src.utils.metrics import evaluate_document_layer_with_labeled_annotations

logger = logging.getLogger(__name__)


def get_type_or_callable(type_str: str) -> Union[Type, Callable]:
    return _resolve_target(type_str, full_key="")


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
        "--document_type",
        type=get_type_or_callable,
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
        "--exclude_annotation_fields",
        type=str,
        nargs="+",
        default=["score"],
        help="annotation fields to exclude from evaluation",
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

        metric_values = evaluate_document_layer_with_labeled_annotations(
            path_or_documents=documents,
            layer=args.layer,
            exclude_labels=args.exclude_labels,
            exclude_annotation_fields=args.exclude_annotation_fields,
        )
        all_metric_values.append(pd.DataFrame(metric_values).T)

    if len(all_metric_values) > 1:
        # mean and stddev over all metric results
        grouped_metric_values = pd.concat(all_metric_values).groupby(level=0)
        logger.info(f"aggregated results (n={len(all_metric_values)}):")
        logger.info(f"\nmean:\n{grouped_metric_values.mean().round(3).to_markdown()}")
        logger.info(f"\nstddev:\n{grouped_metric_values.std().round(3).to_markdown()}")
