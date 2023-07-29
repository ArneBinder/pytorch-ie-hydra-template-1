import dataclasses
from typing import Optional

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span, _post_init_single_label
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

# =========================== Annotation Types ============================= #


@dataclasses.dataclass(eq=True, frozen=True)
class Attribute(Annotation):
    target_annotation: Annotation
    label: str
    value: Optional[str] = None
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)


# ============================= Document Types ============================= #


@dataclasses.dataclass
class TokenDocumentWithEntitiesAndRelations(TokenBasedDocument):
    entities: AnnotationList[Span] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenDocumentWithLabeledEntitiesAndRelations(TokenBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithEntityMentions(TextBasedDocument):
    entity_mentions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[Span] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndRelations(TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class DocumentWithEntitiesRelationsAndLabeledPartitions(TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class BratDocument(TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributions: AnnotationList[Attribute] = annotation_field(target="spans")
    relation_attributions: AnnotationList[Attribute] = annotation_field(target="relations")
