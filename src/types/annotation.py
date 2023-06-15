import dataclasses
from typing import Optional

from pytorch_ie.annotations import _post_init_single_label
from pytorch_ie.core import Annotation


@dataclasses.dataclass(eq=True, frozen=True)
class GeneralBinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclasses.dataclass(eq=True, frozen=True)
class Attribution(Annotation):
    target_annotation: Annotation
    label: str
    value: Optional[str] = None
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)
