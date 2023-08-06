from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from pytorch_ie.annotations import LabeledSpan, Span

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

    entity_map: Dict[int, Span] = {}
    entities_have_labels = False
    for entity in doc[entity_layer]:
        char_start = token_offsets[entity.start][0]
        char_end = token_offsets[entity.end - 1][1]
        char_offset_entity = entity.copy(start=char_start, end=char_end)
        if isinstance(entity, LabeledSpan):
            entities_have_labels = True
        entity_map[entity._id] = char_offset_entity

    if entities_have_labels:
        new_doc = TextDocumentWithLabeledEntitiesAndRelations(
            text=text, id=doc.id, metadata=deepcopy(doc.metadata)
        )
    else:
        new_doc = TextDocumentWithEntitiesAndRelations(
            text=text, id=doc.id, metadata=deepcopy(doc.metadata)
        )
    new_doc.entities.extend(entity_map.values())
    new_doc.add_all_annotations_from_other(
        doc, override_annotation_mapping={"entities": entity_map}
    )
    return new_doc
