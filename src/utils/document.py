from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Match, Optional, Tuple, TypeVar

from asciidag.graph import Graph
from asciidag.node import Node
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document

from src.utils.span import distance, is_contained_in

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def _construct_nodes(name: str, ancestor_graph: Dict[str, List[str]], store: Dict[str, Node]):
    if name in store:
        return store[name]
    deps = ancestor_graph.get(name, [])
    dep_nodes = [_construct_nodes(dep_name, ancestor_graph, store=store) for dep_name in deps]
    store[name] = Node(name, parents=dep_nodes)
    return store[name]


def _revert_edges(edges: Dict[str, List[str]]) -> Dict[str, List[str]]:
    reverted_edges = defaultdict(list)
    for source, targets in edges.items():
        for target in targets:
            reverted_edges[target].append(source)
    return dict(reverted_edges)


def print_document_annotation_graph(
    annotation_graph: Dict[str, List[str]],
    add_root_node: Optional[str] = None,
    remove_node: Optional[str] = None,
    swap_edges: bool = True,
):
    """example call:

    print_document_annotation_graph(
        annotation_graph=document._annotation_graph, remove_node="_artificial_root",
    )

    Args:
        annotation_graph: the annotation graph
        add_root_node: if available, add an artificial root node with this name that connects to all original roots
        remove_node: if available, remove the node with this name from the annotation_graph
        swap_edges: iff True, swap the edges of the graph
    """
    dependency_graph = dict(annotation_graph)
    if remove_node is not None:
        del dependency_graph[remove_node]

    reverted_dependency_graph = _revert_edges(edges=dependency_graph)
    sources = set(dependency_graph) - set(reverted_dependency_graph)
    sinks = set(reverted_dependency_graph) - set(dependency_graph)

    if swap_edges:
        ancestor_graph = dict(reverted_dependency_graph)
        roots = sinks
    else:
        ancestor_graph = dict(dependency_graph)
        roots = sources

    if add_root_node is not None:
        ancestor_graph[add_root_node] = list(roots)
        roots = {add_root_node}

    graph = Graph()
    node_store: Dict[str, Node] = dict()
    root_nodes = [
        _construct_nodes(root, ancestor_graph=ancestor_graph, store=node_store) for root in roots
    ]
    graph.show_nodes(tips=root_nodes)


def create_regex_matcher(pattern):
    return re.compile(pattern).finditer


def _get_partitions_with_matcher(
    text: str,
    matcher_or_pattern: Callable[[str], Iterable[Match]] | str,
    label_group_id: int | None = None,  # = 1,
    label_whitelist: list[str] | None = None,
    skip_initial_partition: bool = False,  # = True
    default_partition_label: str = "partition",
    initial_partition_label: str | None = None,
) -> Iterator[LabeledSpan]:
    """This method yields LabeledSpans as partitions of the given text. matcher is used to search
    for a pattern in the text. If the pattern is found, it returns a Match object that contains
    matched groups. A partition is then created using a span in the matched groups. The span of a
    partition starts from the first match (inclusive) and ends at the next match (exclusive) or at
    the end of the text. A partition is labeled either using the default_partition_label or using
    the list of labels available in label_whitelist. It should be noted that none of the partitions
    overlap.

    :param text: A text that is to be partitioned
    :param matcher_or_pattern: A method or a string. In the former case, that method is used to find a pattern in the text
                    and return an iterator yielding the Match objects, e.g. re.compile(PATTERN).finditer. In the latter,
                    the string is used as a pattern to find the matches in the text.
    :param label_group_id: An integer value (default:None) to select the desired match group from the Match object.
    This match group is then used to create a label for the partition.
    :param label_whitelist: An optional list of labels (default:None) which are allowed to form a partition if
    label_group_id is not None. label_whitelist is the whitelist for the labels created using label_group_id.
                            If label_whitelist is None, then all the labels created using label_group_id will form a
                            partition.
    :param skip_initial_partition: A boolean value (default:False) that prevents the initial partition to be saved.
    :param default_partition_label: A string value (default:partition) to be used as the default label for the parts
    if no label_group_id for the match object is provided.
    :param initial_partition_label: A string value (default:None) to be used as a label for the initial
                                partition. This is only used when skip_initial_partition is False. If it is None then
                                default_partition_label is used as initial_partition_label.
    """
    if isinstance(matcher_or_pattern, str):
        matcher = create_regex_matcher(matcher_or_pattern)
    else:
        matcher = matcher_or_pattern
    if initial_partition_label is None:
        initial_partition_label = default_partition_label
    previous_start = previous_label = None
    if not skip_initial_partition:
        if label_whitelist is None or initial_partition_label in label_whitelist:
            previous_start = 0
            previous_label = initial_partition_label
    for match in matcher(text):
        if label_group_id is not None:
            start = match.start(label_group_id)
            end = match.end(label_group_id)
            label = text[start:end]
        else:
            label = default_partition_label
        if label_whitelist is None or label in label_whitelist:
            if previous_start is not None and previous_label is not None:
                end = match.start()
                span = LabeledSpan(start=previous_start, end=end, label=previous_label)
                yield span

            previous_start = match.start()
            previous_label = label

    if previous_start is not None and previous_label is not None:
        end = len(text)
        span = LabeledSpan(start=previous_start, end=end, label=previous_label)
        yield span


def add_partitions_with_regex(
    doc: D,
    text_field_name: str = "text",
    partition_layer_name: str = "partitions",
    inplace: bool = False,
    **partition_kwargs,
) -> D:
    if not inplace:
        doc = type(doc).fromdict(doc.asdict())
    text: str = getattr(doc, text_field_name)
    partition_annotations = list(_get_partitions_with_matcher(text=text, **partition_kwargs))
    doc[partition_layer_name].extend(partition_annotations)
    return doc


def _remove_overlapping_entities(
    entities: Iterable[Dict[str, Any]], relations: Iterable[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sorted_entities = sorted(entities, key=lambda span: span["start"])
    entities_wo_overlap = []
    skipped_entities = []
    last_end = 0
    for entity_dict in sorted_entities:
        if entity_dict["start"] < last_end:
            skipped_entities.append(entity_dict)
        else:
            entities_wo_overlap.append(entity_dict)
            last_end = entity_dict["end"]
    if len(skipped_entities) > 0:
        logger.warning(f"skipped overlapping entities: {skipped_entities}")
    valid_entity_ids = set(entity_dict["_id"] for entity_dict in entities_wo_overlap)
    valid_relations = [
        relation_dict
        for relation_dict in relations
        if relation_dict["head"] in valid_entity_ids and relation_dict["tail"] in valid_entity_ids
    ]
    return entities_wo_overlap, valid_relations


def remove_overlapping_entities(
    doc: D,
    entity_layer_name: str = "entities",
    relation_layer_name: str = "relations",
) -> D:
    document_dict = doc.asdict()
    entities_wo_overlap, valid_relations = _remove_overlapping_entities(
        entities=document_dict[entity_layer_name]["annotations"],
        relations=document_dict[relation_layer_name]["annotations"],
    )

    document_dict[entity_layer_name] = {
        "annotations": entities_wo_overlap,
        "predictions": [],
    }
    document_dict[relation_layer_name] = {
        "annotations": valid_relations,
        "predictions": [],
    }
    new_doc = type(doc).fromdict(document_dict)

    return new_doc


def clear_annotation_layers(
    doc: D, layer_names: List[str], predictions: bool = False, inplace: bool = False
) -> D:
    if not inplace:
        doc = type(doc).fromdict(doc.asdict())
    for layer_name in layer_names:
        if predictions:
            doc[layer_name].predictions.clear()
        else:
            doc[layer_name].clear()
    return doc


def move_annotations_from_predictions(doc: D, layer_names: List[str], inplace: bool = False) -> D:
    if not inplace:
        doc = type(doc).fromdict(doc.asdict())
    for layer_name in layer_names:
        annotations = list(doc[layer_name].predictions)
        # remove any previous annotations
        doc[layer_name].clear()
        # each annotation can be attached to just one annotation container, so we need to clear the predictions
        doc[layer_name].predictions.clear()
        doc[layer_name].extend(annotations)
    return doc


def move_annotations_to_predictions(doc: D, layer_names: List[str], inplace: bool = False) -> D:
    if not inplace:
        doc = type(doc).fromdict(doc.asdict())
    for layer_name in layer_names:
        annotations = list(doc[layer_name])
        # each annotation can be attached to just one annotation container, so we need to clear the layer
        doc[layer_name].clear()
        # remove any previous annotations
        doc[layer_name].predictions.clear()
        doc[layer_name].predictions.extend(annotations)
    return doc


def add_annotations_from_other_documents(
    docs: List[D],
    other_docs: List[Document],
    layer_names: List[str],
    from_predictions: bool = False,
    to_predictions: bool = False,
    clear_before: bool = True,
    inplace: bool = False,
) -> List[D]:
    prepared_documents = []
    for i, doc in enumerate(docs):
        if not inplace:
            doc = type(doc).fromdict(doc.asdict())
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
        prepared_documents.append(doc)
    return prepared_documents


def trim_spans(
    document: D,
    span_layer: str,
    text_field: str = "text",
    relation_layer: Optional[str] = None,
    skip_empty: bool = True,
    inplace: bool = False,
) -> D:
    """Remove the whitespace at the beginning and end of span annotations. If a relation layer is
    given, the relations will be updated to point to the new spans.

    Args:
        document: The document to trim its span annotations.
        span_layer: The name of the span layer to trim.
        text_field: The name of the text field in the document.
        relation_layer: The name of the relation layer to update. If None, the relations will not be updated.
        skip_empty: If True, empty spans will be skipped. Otherwise, an error will be raised.
        copy: If True, the document will be copied before trimming.

    Returns:
        The document with trimmed spans.
    """
    if not inplace:
        document = type(document).fromdict(document.asdict())

    spans: AnnotationList[LabeledSpan] = document[span_layer]
    old2new_spans = {}
    text = getattr(document, text_field)
    for span in spans:
        span_text = text[span.start : span.end]
        new_start = span.start + len(span_text) - len(span_text.lstrip())
        new_end = span.end - len(span_text) + len(span_text.rstrip())
        # if the new span is empty and skip_empty is True, skip it
        if new_start < new_end or not skip_empty:
            # if skip_empty is False and the new span is empty, log a warning and create a new zero-width span
            # by using the old start position
            if new_start >= new_end:
                logger.warning(
                    f"Span {span} is empty after trimming. {'Skipping it.' if skip_empty else ''}"
                )
                new_start = span.start
                new_end = span.start
            new_span = LabeledSpan(
                start=new_start,
                end=new_end,
                label=span.label,
                score=span.score,
            )
            old2new_spans[span] = new_span

    spans.clear()
    spans.extend(old2new_spans.values())

    if relation_layer is not None:
        relations: AnnotationList[BinaryRelation] = document[relation_layer]
        new_relations = []
        for relation in relations:
            if relation.head not in old2new_spans or relation.tail not in old2new_spans:
                logger.warning(
                    f"Relation {relation} is removed because one of its spans was removed."
                )
                continue
            head = old2new_spans[relation.head]
            tail = old2new_spans[relation.tail]
            new_relations.append(
                BinaryRelation(
                    head=head,
                    tail=tail,
                    label=relation.label,
                    score=relation.score,
                )
            )
        relations.clear()
        relations.extend(new_relations)

    return document


def add_reversed_relations(
    document: D,
    relation_layer: str = "relations",
    symmetric_relation_labels: Optional[List[str]] = None,
    label_suffix: str = "_reversed",
    allow_already_reversed_relations: bool = False,
    use_predictions: bool = False,
    inplace: bool = False,
) -> D:
    if not inplace:
        document = type(document).fromdict(document.asdict())
    symmetric_relation_labels = symmetric_relation_labels or []

    rel_layer = document[relation_layer]
    if use_predictions:
        rel_layer = rel_layer.predictions

    # get all relations before adding any reversed
    available_relations = {(rel.head, rel.tail): rel for rel in rel_layer}
    for rel in list(rel_layer):
        new_label = (
            rel.label if rel.label in symmetric_relation_labels else f"{rel.label}{label_suffix}"
        )
        new_relation = BinaryRelation(label=new_label, head=rel.tail, tail=rel.head)
        if (new_relation.head, new_relation.tail) in available_relations:
            # If an entity pair of reversed relation is present in the available relations then we check if we want
            # to allow already existing reversed relations or not. If we allow then we do not add the reversed
            # relation to the document but move on to the next relation otherwise we generate a LookupError
            # exception.
            if allow_already_reversed_relations:
                continue
            else:
                raise LookupError(
                    f"Entity pair of new relation ({new_relation}) already belongs to a relation: "
                    f"{available_relations[(new_relation.head, new_relation.tail)]}"
                )
        else:
            rel_layer.append(new_relation)

    return document


def get_single_target_layer(document: Document, layer: AnnotationList):
    if len(layer._targets) != 1:
        raise Exception(
            f"the layer is expected to have exactly one target layer, but it has "
            f"the following targets: {layer._targets}"
        )
    target_layer_name = layer._targets[0]
    return document[target_layer_name]


def add_candidate_relations(
    document: D,
    label: str = "no_relation",
    relation_layer: str = "relations",
    use_predictions: bool = False,
    partition_layer: Optional[str] = None,
    max_distance: Optional[int] = None,
    distance_type: str = "inner",
    sort_by_distance: bool = True,
    n_max: Optional[int] = None,
    # this should not be used during prediction, because it will leak gold relation information!
    n_max_factor: Optional[float] = None,
    inplace: bool = True,
) -> D:
    if not inplace:
        document = type(document).fromdict(document.asdict())
    rel_layer = document[relation_layer]
    if use_predictions:
        rel_layer = rel_layer.predictions
    available_relation_mapping = {(rel.head, rel.tail): rel for rel in rel_layer}
    if partition_layer is not None:
        available_partitions = document[partition_layer]
    else:
        available_partitions = [None]
    entity_layer = get_single_target_layer(document=document, layer=rel_layer)
    if use_predictions:
        entity_layer = entity_layer.predictions

    candidates_with_distance = {}
    distances_taken = defaultdict(list)
    num_relations_in_partition = 0
    available_rels_within_allowed_distance = set()
    for partition in available_partitions:
        if partition is not None:
            available_entities = [
                entity
                for entity in entity_layer
                if is_contained_in((entity.start, entity.end), (partition.start, partition.end))
            ]
        else:
            available_entities = list(entity_layer)
        for head in available_entities:
            for tail in available_entities:
                if head == tail:
                    continue
                d = distance(
                    (head.start, head.end),
                    (tail.start, tail.end),
                    distance_type,
                )
                if max_distance is not None and d > max_distance:
                    continue
                if (head, tail) in available_relation_mapping:
                    num_relations_in_partition += 1
                    distances_taken[available_relation_mapping[(head, tail)].label].append(d)
                    available_rels_within_allowed_distance.add(
                        available_relation_mapping[(head, tail)]
                    )
                    continue
                candidates_with_distance[(head, tail)] = d
    if sort_by_distance:
        candidates_with_distance_list = sorted(
            candidates_with_distance.items(), key=lambda item: item[1]
        )
    else:
        candidates_with_distance_list = list(candidates_with_distance.items())
        random.shuffle(candidates_with_distance_list)
    n_added = 0
    if n_max is not None:
        candidates_with_distance_list = candidates_with_distance_list[:n_max]
    if n_max_factor is not None:
        n_max_by_factor = int(len(rel_layer) * n_max_factor)
        candidates_with_distance_list = candidates_with_distance_list[:n_max_by_factor]
    # num_total_candidates = len(entities) * len(entities) - len(entities)
    # update_statistics("num_total_relation_candidates", num_total_candidates)
    # num_available_relations = len(rel_layer)
    # update_statistics("num_available_relations", num_available_relations)
    for (head, tail), d in candidates_with_distance_list:
        new_relation = BinaryRelation(label=label, head=head, tail=tail)
        rel_layer.append(new_relation)
        distances_taken[label].append(d)
        n_added += 1

    return document
