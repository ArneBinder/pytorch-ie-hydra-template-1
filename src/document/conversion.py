from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span

from src.document.types import (
    TextDocumentWithEntitiesAndRelations,
    TextDocumentWithLabeledEntitiesAndRelations,
    TokenDocumentWithEntitiesAndRelations,
)

logger = logging.getLogger(__name__)


def token_based_document_with_entities_and_relations_to_text_based(
    doc: TokenDocumentWithEntitiesAndRelations,
    token_field: str = "tokens",
    entity_layer: str = "entities",
    relation_layer: str = "relations",
    token_separator: str = " ",
) -> Union[TextDocumentWithEntitiesAndRelations, TextDocumentWithLabeledEntitiesAndRelations]:
    start = 0
    token_offsets: List[Tuple[int, int]] = []
    tokens = getattr(doc, token_field)
    for token in tokens:
        end = start + len(token)
        token_offsets.append((start, end))
        # we add the separator after each token
        start = end + len(token_separator)

    text = token_separator.join([token for token in tokens])

    entity_map: Dict[Tuple[int, int], Span] = {}

    entities_have_labels = False
    for entity in doc[entity_layer]:
        char_start = token_offsets[entity.start][0]
        char_end = token_offsets[entity.end - 1][1]
        if isinstance(entity, LabeledSpan):
            char_offset_entity = LabeledSpan(start=char_start, end=char_end, label=entity.label)
            entities_have_labels = True
        else:
            char_offset_entity = Span(start=char_start, end=char_end)
        entity_map[(entity.start, entity.end)] = char_offset_entity

    relations: List[BinaryRelation] = []
    for relation in doc[relation_layer]:
        relations.append(
            BinaryRelation(
                head=entity_map[(relation.head.start, relation.head.end)],
                tail=entity_map[(relation.tail.start, relation.tail.end)],
                label=relation.label,
            )
        )

    if entities_have_labels:
        new_doc = TextDocumentWithLabeledEntitiesAndRelations(
            text=text, id=doc.id, metadata=deepcopy(doc.metadata)
        )
    else:
        new_doc = TextDocumentWithEntitiesAndRelations(
            text=text, id=doc.id, metadata=deepcopy(doc.metadata)
        )
    new_doc.entities.extend(entity_map.values())
    new_doc.relations.extend(relations)
    return new_doc
