import timeit
from collections.abc import Iterable, Sequence
from typing import Any, Dict, Optional, Union

from pie_core import AnnotationPipeline, Document

from src.serializer.interface import DocumentSerializer
from src.utils.logging_utils import get_pylogger

log = get_pylogger(__name__)


def document_batch_iter(
    dataset: Iterable[Document], batch_size: int
) -> Iterable[Sequence[Document]]:
    if isinstance(dataset, Sequence):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]
    elif isinstance(dataset, Iterable):
        docs = []
        for doc in dataset:
            docs.append(doc)
            if len(docs) == batch_size:
                yield docs
                docs = []
        if docs:
            yield docs
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def predict_and_serialize(
    pipeline: Optional[AnnotationPipeline],
    serializer: Optional[DocumentSerializer],
    dataset: Iterable[Document],
    document_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if pipeline is not None:
        log.info("Starting inference!")
        prediction_time = 0.0
    else:
        log.warning("No prediction pipeline is defined, skip inference!")
        prediction_time = None
    docs_batch: Union[Iterable[Document], Sequence[Document]]

    batch_iter: Union[Sequence[Iterable[Document]], Iterable[Sequence[Document]]]
    if document_batch_size is None:
        batch_iter = [dataset]
    else:
        batch_iter = document_batch_iter(dataset=dataset, batch_size=document_batch_size)

    append = False
    for docs_batch in batch_iter:
        if pipeline is not None:
            t_start = timeit.default_timer()
            docs_batch = pipeline(docs_batch, inplace=False)
            prediction_time += timeit.default_timer() - t_start  # type: ignore

        # serialize the documents
        if serializer is not None:
            # the serializer should not return the serialized documents, but write them to disk
            # and instead return some metadata such as the path to the serialized documents
            serializer_result = serializer(docs_batch, append=append)
            if "serializer" in result and result["serializer"] != serializer_result:
                log.warning(
                    f"serializer result changed from {result['serializer']} to {serializer_result}"
                    " during prediction. Only the last result is returned."
                )
            result["serializer"] = serializer_result
            append = True

    if prediction_time is not None:
        result["prediction_time"] = prediction_time

    return result
