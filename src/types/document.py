import dataclasses
from typing import Any, Dict, Optional, Tuple

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, Document, annotation_field

from src.types.annotation import Attribution


@dataclasses.dataclass
class DocumentBase:
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TokenBasedDocument(Document):
    tokens: Tuple[str, ...]


@dataclasses.dataclass
class TextBasedDocument(Document):
    text: str


@dataclasses.dataclass
class TokenDocumentWithEntitiesAndRelations(DocumentBase, TokenBasedDocument):
    entities: AnnotationList[Span] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenDocumentWithLabeledEntitiesAndRelations(DocumentBase, TokenBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithEntityMentions(DocumentBase, TextBasedDocument):
    entity_mentions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(DocumentBase, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[Span] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndRelations(DocumentBase, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class DocumentWithEntitiesRelationsAndLabeledPartitions(DocumentBase, TextBasedDocument):
    """Possible input class for TransformerRETextClassificationTaskModule."""

    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class BratDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    span_attributions: AnnotationList[Attribution] = annotation_field(target="spans")
    relation_attributions: AnnotationList[Attribution] = annotation_field(target="relations")
