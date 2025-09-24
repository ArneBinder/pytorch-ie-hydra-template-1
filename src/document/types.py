import dataclasses
from typing import Optional

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import Annotation, AnnotationLayer, annotation_field
from pytorch_ie.documents import TextBasedDocument

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
        result = f"label={self.label}"
        if self.is_attached:
            result += f",annotation={self.annotation}"
        if self.type is not None:
            result += f",type={self.type}"
        if self.score is not None:
            result += f",score={self.score}"
        return f"{self.__class__.__name__}({result})"


# ============================= Document Types ============================= #


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndEntityAttributes(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entity_attributes: AnnotationLayer[Attribute] = annotation_field(target="entities")
