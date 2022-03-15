from typing import Any, Dict

from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule


def additional_model_kwargs(taskmodule: TransformerTokenClassificationTaskModule) -> Dict[str, Any]:
    return dict(num_classes=len(taskmodule.label_to_id))
