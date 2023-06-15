from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

from pytorch_ie import AutoPipeline
from pytorch_ie.core import Document

from src.utils.document import (
    add_annotations_from_other_documents,
    add_candidate_relations,
    clear_annotation_layers,
    move_annotations_from_predictions,
    move_annotations_to_predictions,
)


def process_pipeline_steps(
    documents: Sequence[Document],
    processors: Dict[str, Callable[[Document], Optional[Document]]],
    inplace: bool = False,
):
    if not inplace:
        documents = [type(doc).fromdict(doc.asdict()) for doc in documents]

    # do the actual inference
    for step_name, processor in processors.items():
        print(f"process {step_name} ...")
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
        device: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
        **processor_kwargs,
    ):
        self.ner_model_path = ner_model_path
        self.re_model_path = re_model_path
        self.processor_kwargs = processor_kwargs or {}
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

    def __call__(self, documents: Sequence[Document], inplace: bool = False):

        if not inplace:
            documents = [type(doc).fromdict(doc.asdict()) for doc in documents]

        docs_with_predictions = process_pipeline_steps(
            documents=documents,
            processors={
                "clear_annotations": partial(
                    process_documents,
                    processor=clear_annotation_layers,
                    layer_names=["entities", "relations"],
                    **self.processor_kwargs.get("clear_annotations", {}),
                ),
                "ner_pipeline": AutoPipeline.from_pretrained(
                    self.ner_model_path, **self.processor_kwargs.get("ner_pipeline", {})
                ),
                "use_predicted_entities": partial(
                    process_documents,
                    processor=move_annotations_from_predictions,
                    layer_names=["entities"],
                    **self.processor_kwargs.get("use_predicted_entities", {}),
                ),
                "create_candidate_relations": partial(
                    process_documents,
                    processor=add_candidate_relations,
                    **self.processor_kwargs.get("create_candidate_relations", {}),
                ),
                "re_pipeline": AutoPipeline.from_pretrained(
                    self.re_model_path, **self.processor_kwargs.get("re_pipeline", {})
                ),
                # otherwise we can not move the entities back to predictions
                "clear_candidate_relations": partial(
                    process_documents,
                    processor=clear_annotation_layers,
                    layer_names=["relations"],
                    **self.processor_kwargs.get("clear_candidate_relations", {}),
                ),
                "move_entities_to_predictions": partial(
                    process_documents,
                    processor=move_annotations_to_predictions,
                    layer_names=["entities"],
                    **self.processor_kwargs.get("move_entities_to_predictions", {}),
                ),
                "re_add_gold_data": partial(
                    add_annotations_from_other_documents,
                    other_docs=documents,
                    layer_names=["entities", "relations"],
                    **self.processor_kwargs.get("re_add_gold_data", {}),
                ),
            },
        )
        return docs_with_predictions
