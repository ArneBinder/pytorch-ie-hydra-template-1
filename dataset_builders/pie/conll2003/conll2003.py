from dataclasses import dataclass
from typing import List, Sequence, Tuple

import datasets
from pie_core import AnnotationLayer, annotation_field
from pie_datasets import GeneratorBasedBuilder
from pie_documents.annotations import LabeledSpan
from pie_documents.documents import TextBasedDocument, TextDocumentWithLabeledSpans
from pie_documents.utils.sequence_tagging import tag_sequence_to_token_spans


def tokens_and_tags_to_text_and_labeled_spans(
    tokens: Sequence[str], tags: Sequence[str]
) -> Tuple[str, Sequence[LabeledSpan]]:
    start = 0
    token_offsets: List[Tuple[int, int]] = []
    for token in tokens:
        end = start + len(token)
        token_offsets.append((start, end))
        # we add a space after each token
        start = end + 1

    text = " ".join(tokens)

    spans: List[LabeledSpan] = []
    for label, (start, end) in tag_sequence_to_token_spans(tag_sequence=tags):
        spans.append(
            LabeledSpan(start=token_offsets[start][0], end=token_offsets[end][1], label=label)
        )

    return text, spans


@dataclass
class CoNLL2003Document(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


class Conll2003(GeneratorBasedBuilder):
    DOCUMENT_TYPE = CoNLL2003Document

    BASE_DATASET_PATH = "conll2003"
    BASE_DATASET_REVISION = "01ad4ad271976c5258b9ed9b910469a806ff3288"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="conll2003", version=datasets.Version("1.0.0"), description="CoNLL2003 dataset"
        ),
    ]

    DOCUMENT_CONVERTERS = {
        TextDocumentWithLabeledSpans: {
            # just rename the layer
            "entities": "labeled_spans",
        }
    }

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = CoNLL2003Document(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
