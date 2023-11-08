import dataclasses
from typing import Optional

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

# =========================== Annotation Types ============================= #


@dataclasses.dataclass(eq=True, frozen=True)
class Attribute(Annotation):
    annotation: Annotation
    label: str
    type: Optional[str] = None
    score: Optional[float] = dataclasses.field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise ValueError("label must be a single string.")
        if not (self.score is None or isinstance(self.score, float)):
            raise ValueError("score must be a single float.")

    def __str__(self) -> str:
        if self.target is not None:
            result = f"label={self.label},annotation={self.annotation}"
        else:
            result = f"label={self.label}"
        if self.type is not None:
            result += f",type={self.type}"
        if self.score is not None:
            result += f",score={self.score}"
        return f"{self.__class__.__name__}({result})"


# ============================= Document Types ============================= #


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndEntityAttributes(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    entity_attributes: AnnotationList[Attribute] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TokenDocumentWithLabeledSpansAndBinaryRelations
):
    labeled_partitions: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
