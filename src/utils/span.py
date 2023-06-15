from typing import Tuple


def get_overlap_len(indices_1: Tuple[int, int], indices_2: Tuple[int, int]) -> int:
    if indices_1[0] > indices_2[0]:
        tmp = indices_1
        indices_1 = indices_2
        indices_2 = tmp
    if indices_1[1] <= indices_2[0]:
        return 0
    return min(indices_1[1] - indices_2[0], indices_2[1] - indices_2[0])


def have_overlap(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    other_start_overlaps = start_end[0] <= other_start_end[0] < start_end[1]
    other_end_overlaps = start_end[0] < other_start_end[1] <= start_end[1]
    start_overlaps_other = other_start_end[0] <= start_end[0] < other_start_end[1]
    end_overlaps_other = other_start_end[0] < start_end[1] <= other_start_end[1]
    return other_start_overlaps or other_end_overlaps or start_overlaps_other or end_overlaps_other


def is_contained_in(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    return other_start_end[0] <= start_end[0] and start_end[1] <= other_start_end[1]


def distance_center(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> float:
    return abs((start_end[0] + start_end[1]) / 2 - (other_start_end[0] + other_start_end[1]) / 2)


def distance_outer(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> float:
    _max = max(start_end[0], start_end[1], other_start_end[0], other_start_end[1])
    _min = min(start_end[0], start_end[1], other_start_end[0], other_start_end[1])
    return _max - _min


def distance_inner(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> float:
    assert not have_overlap(
        start_end, other_start_end
    ), "can not calculate inner span distance for overlapping spans"
    if start_end[0] < other_start_end[0]:
        return other_start_end[0] - start_end[1]
    else:
        return start_end[0] - other_start_end[1]


def distance(start_end: Tuple[int, int], other_start_end: Tuple[int, int], distance_type: str):
    if distance_type == "center":
        return distance_center(start_end, other_start_end)
    elif distance_type == "inner":
        return distance_inner(start_end, other_start_end)
    elif distance_type == "outer":
        return distance_outer(start_end, other_start_end)
    else:
        raise ValueError(
            f"unknown distance_type={distance_type}. use one of: center, inner, outer"
        )
