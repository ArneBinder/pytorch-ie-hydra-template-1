import json
import logging
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Callable, Collection, Dict, Hashable, List, Optional, Set, Tuple, Type, Union

import pandas as pd
from pytorch_ie.core import Annotation, Document

from src.document.types import DocumentWithEntitiesRelationsAndLabeledPartitions
from src.serializer import JsonSerializer

logger = logging.getLogger(__name__)


def eval_counts_for_layer(
    document: Document,
    layer: str,
    annotation_filter: Optional[Callable[[Annotation], bool]] = None,
    annotation_mapper: Optional[Callable[[Annotation], Hashable]] = None,
) -> Tuple[int, int, int]:
    annotation_filter = annotation_filter or (lambda ann: True)
    annotation_mapper = annotation_mapper or (lambda ann: ann)
    predicted_annotations = set(
        annotation_mapper(ann) for ann in document[layer].predictions if annotation_filter(ann)
    )
    gold_annotations = set(
        annotation_mapper(ann) for ann in document[layer] if annotation_filter(ann)
    )
    tp = len([ann for ann in predicted_annotations & gold_annotations])
    fn = len([ann for ann in gold_annotations - predicted_annotations])
    fp = len([ann for ann in predicted_annotations - gold_annotations])
    return tp, fp, fn


def _remove_annotation_fields(ann: Annotation, exclude_annotation_fields: Set[str]):
    return json.dumps(
        {k: v for k, v in ann.asdict().items() if k not in exclude_annotation_fields},  # type: ignore
        sort_keys=True,
    )


class F1MetricForLabeledAnnotations:
    def __init__(
        self,
        layer: str,
        labels: Optional[Collection[str]] = None,
        exclude_labels: Optional[Collection[str]] = None,
        exclude_annotation_fields: Optional[List[str]] = None,
    ):
        self.layer = layer
        self.exclude_labels = set(exclude_labels or [])
        self.labels = set(labels or []) - self.exclude_labels
        assert "MICRO" not in self.labels and "MACRO" not in self.labels

        self.annotation_mapper: Optional[Callable[[Annotation], Hashable]] = None
        if exclude_annotation_fields is not None:
            exclude_annotation_fields.append("_id")
            self.annotation_mapper = partial(
                _remove_annotation_fields, exclude_annotation_fields=set(exclude_annotation_fields)
            )

        self.reset()

    def _add_counts(self, counts: Tuple[int, int, int], label: str):
        self.counts[label] = (
            self.counts[label][0] + counts[0],
            self.counts[label][1] + counts[1],
            self.counts[label][2] + counts[2],
        )

    def __call__(self, document: Union[List[Document], Document]):
        if isinstance(document, list):
            for doc in document:
                self(doc)
        elif isinstance(document, Document):
            new_counts = eval_counts_for_layer(
                document=document,
                layer=self.layer,
                annotation_filter=lambda ann: ann.label not in self.exclude_labels,
                annotation_mapper=self.annotation_mapper,
            )
            self._add_counts(new_counts, label="MICRO")
            for label in self.labels:
                new_counts = eval_counts_for_layer(
                    document=document,
                    layer=self.layer,
                    annotation_filter=lambda ann: ann.label == label,
                    annotation_mapper=self.annotation_mapper,
                )
                self._add_counts(new_counts, label=label)
        else:
            raise Exception(f"document has unknown type: {type(document)}")

    def reset(self):
        self.counts = defaultdict(lambda: (0, 0, 0))

    def values(self, reset: bool = True, show_as_markdown: bool = False):

        res = dict()
        res["MACRO"] = {"f1": 0.0, "p": 0.0, "r": 0.0}
        for label, counts in self.counts.items():
            tp, fp, fn = counts
            if tp == 0:
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            res[label] = {"f1": f1, "p": p, "r": r}
            if label in self.labels:
                res["MACRO"]["f1"] += f1 / len(self.labels)
                res["MACRO"]["p"] += p / len(self.labels)
                res["MACRO"]["r"] += r / len(self.labels)
        if reset:
            self.reset()
        if show_as_markdown:
            logger.info(f"\n{self.layer}:\n{pd.DataFrame(res).round(3).T.to_markdown()}")
        return res


def evaluate_document_layer_with_labeled_annotations(
    path_or_documents: Union[str, List[Document]],
    layer: str,
    document_type: Optional[Type[Document]] = DocumentWithEntitiesRelationsAndLabeledPartitions,
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
    labels = set(chain(*[[ann.label for ann in doc[layer]] for doc in documents]))
    f1metric = F1MetricForLabeledAnnotations(
        layer=layer,
        labels=labels,
        exclude_labels=exclude_labels,
        exclude_annotation_fields=exclude_annotation_fields,
    )
    f1metric(documents)

    metric_values = f1metric.values(show_as_markdown=show_as_markdown)
    return metric_values
