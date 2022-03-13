from pytorch_ie.models import (
    TransformerTokenClassificationModel as PIETransformerTokenClassificationModel,
)
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from pytorch_ie.taskmodules.taskmodule import TaskModule


class TransformerTokenClassificationModel(PIETransformerTokenClassificationModel):
    @classmethod
    def from_taskmodule(cls, taskmodule: TaskModule, **kwargs):
        if isinstance(taskmodule, TransformerTokenClassificationTaskModule):
            return cls(num_classes=len(taskmodule.label_to_id), **kwargs)
        else:
            raise ValueError(f"taskmodule has unknown type: {type(taskmodule)}")
