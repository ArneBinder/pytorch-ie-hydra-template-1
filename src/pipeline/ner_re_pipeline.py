from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, TypeVar

from pytorch_ie import AutoPipeline
from pytorch_ie.core import Document

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def clear_annotation_layers(doc: D, layer_names: List[str], predictions: bool = False) -> None:
    for layer_name in layer_names:
        if predictions:
            doc[layer_name].predictions.clear()
        else:
            doc[layer_name].clear()


def move_annotations_from_predictions(doc: D, layer_names: List[str]) -> None:
    for layer_name in layer_names:
        annotations = list(doc[layer_name].predictions)
        # remove any previous annotations
        doc[layer_name].clear()
        # each annotation can be attached to just one annotation container, so we need to clear the predictions
        doc[layer_name].predictions.clear()
        doc[layer_name].extend(annotations)


def move_annotations_to_predictions(doc: D, layer_names: List[str]) -> None:
    for layer_name in layer_names:
        annotations = list(doc[layer_name])
        # each annotation can be attached to just one annotation container, so we need to clear the layer
        doc[layer_name].clear()
        # remove any previous annotations
        doc[layer_name].predictions.clear()
        doc[layer_name].predictions.extend(annotations)


def add_annotations_from_other_documents(
    docs: List[D],
    other_docs: List[Document],
    layer_names: List[str],
    from_predictions: bool = False,
    to_predictions: bool = False,
    clear_before: bool = True,
) -> None:
    for i, doc in enumerate(docs):
        other_doc = other_docs[i]
        # copy to not modify the input
        other_doc = type(other_doc).fromdict(other_doc.asdict())

        for layer_name in layer_names:
            if clear_before:
                doc[layer_name].clear()
            other_layer = other_doc[layer_name]
            if from_predictions:
                other_layer = other_layer.predictions
            other_annotations = list(other_layer)
            other_layer.clear()
            if to_predictions:
                doc[layer_name].predictions.extend(other_annotations)
            else:
                doc[layer_name].extend(other_annotations)


def process_pipeline_steps(
    documents: Sequence[Document],
    processors: Dict[str, Callable[[Sequence[Document]], Optional[Sequence[Document]]]],
) -> Sequence[Document]:

    # call the processors in the order they are provided
    for step_name, processor in processors.items():
        logger.info(f"process {step_name} ...")
        processed_documents = processor(documents)
        if processed_documents is not None:
            documents = processed_documents

    return documents


def process_documents(
    documents: List[Document], processor: Callable[..., Optional[Document]], **kwargs
) -> List[Document]:
    result = []
    for doc in documents:
        processed_doc = processor(doc, **kwargs)
        if processed_doc is not None:
            result.append(processed_doc)
        else:
            result.append(doc)
    return result


class NerRePipeline:
    def __init__(
        self,
        ner_model_path: str,
        re_model_path: str,
        entity_layer: str,
        relation_layer: str,
        device: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
        **processor_kwargs,
    ):
        self.ner_model_path = ner_model_path
        self.re_model_path = re_model_path
        self.processor_kwargs = processor_kwargs or {}
        self.entity_layer = entity_layer
        self.relation_layer = relation_layer
        # set some values for the inference processors, if provided
        for inference_pipeline in ["ner_pipeline", "re_pipeline"]:
            if inference_pipeline not in self.processor_kwargs:
                self.processor_kwargs[inference_pipeline] = {}
            if "device" not in self.processor_kwargs[inference_pipeline] and device is not None:
                self.processor_kwargs[inference_pipeline]["device"] = device
            if (
                "batch_size" not in self.processor_kwargs[inference_pipeline]
                and batch_size is not None
            ):
                self.processor_kwargs[inference_pipeline]["batch_size"] = batch_size
            if (
                "show_progress_bar" not in self.processor_kwargs[inference_pipeline]
                and show_progress_bar is not None
            ):
                self.processor_kwargs[inference_pipeline]["show_progress_bar"] = show_progress_bar

    def __call__(self, documents: Sequence[Document], inplace: bool = False) -> Sequence[Document]:

        input_docs: Sequence[Document]
        # we need to keep the original documents to add the gold data back
        original_docs: Sequence[Document]
        if inplace:
            input_docs = documents
            original_docs = [doc.copy() for doc in documents]
        else:
            input_docs = [doc.copy() for doc in documents]
            original_docs = documents

        docs_with_predictions = process_pipeline_steps(
            documents=input_docs,
            processors={
                "clear_annotations": partial(
                    process_documents,
                    processor=clear_annotation_layers,
                    layer_names=[self.entity_layer, self.relation_layer],
                    **self.processor_kwargs.get("clear_annotations", {}),
                ),
                "ner_pipeline": AutoPipeline.from_pretrained(
                    self.ner_model_path, **self.processor_kwargs.get("ner_pipeline", {})
                ),
                "use_predicted_entities": partial(
                    process_documents,
                    processor=move_annotations_from_predictions,
                    layer_names=[self.entity_layer],
                    **self.processor_kwargs.get("use_predicted_entities", {}),
                ),
                # "create_candidate_relations": partial(
                #    process_documents,
                #    processor=CandidateRelationAdder(
                #        **self.processor_kwargs.get("create_candidate_relations", {})
                #    ),
                # ),
                "re_pipeline": AutoPipeline.from_pretrained(
                    self.re_model_path, **self.processor_kwargs.get("re_pipeline", {})
                ),
                # otherwise we can not move the entities back to predictions
                "clear_candidate_relations": partial(
                    process_documents,
                    processor=clear_annotation_layers,
                    layer_names=[self.relation_layer],
                    **self.processor_kwargs.get("clear_candidate_relations", {}),
                ),
                "move_entities_to_predictions": partial(
                    process_documents,
                    processor=move_annotations_to_predictions,
                    layer_names=[self.entity_layer],
                    **self.processor_kwargs.get("move_entities_to_predictions", {}),
                ),
                "re_add_gold_data": partial(
                    add_annotations_from_other_documents,
                    other_docs=original_docs,
                    layer_names=[self.entity_layer, self.relation_layer],
                    **self.processor_kwargs.get("re_add_gold_data", {}),
                ),
            },
        )
        return docs_with_predictions
