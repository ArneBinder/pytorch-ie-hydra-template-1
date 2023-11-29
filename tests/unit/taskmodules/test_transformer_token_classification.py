import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
)
from transformers import BatchEncoding

from src.taskmodules import MyTokenClassificationTaskModule


def _config_to_str(cfg: Dict[str, Any]) -> str:
    # Converts a configuration dictionary to a string representation
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIG_DEFAULT = {}
CONFIG_MAX_WINDOW = {
    "tokenize_kwargs": {"max_length": 8, "truncation": True, "return_overflowing_tokens": True}
}
CONFIG_MAX_WINDOW_WITH_STRIDE = {
    "tokenize_kwargs": {
        "max_length": 8,
        "stride": 2,
        "truncation": True,
        "return_overflowing_tokens": True,
    }
}
CONFIG_PARTITIONS = {"partition_annotation": "sentences"}

CONFIGS: List[Dict[str, Any]] = [
    CONFIG_DEFAULT,
    CONFIG_MAX_WINDOW,
    CONFIG_MAX_WINDOW_WITH_STRIDE,
    CONFIG_PARTITIONS,
]

CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    """
    - Provides clean and readable test configurations.
    - Yields config dictionaries from the CONFIGS list to produce clean test case identifiers.

    """
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def config_str(config):
    # Fixture returning a string representation of the config
    return _config_to_str(config)


@pytest.fixture(scope="module")
def unprepared_taskmodule(config):
    """
    - Prepares a task module with the specified tokenizer and configuration.
    - Sets up the task module with a unprepared state for testing purposes.

    """
    return MyTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", span_annotation="entities", **config
    )


@dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    sentences: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.fixture(scope="module")
def documents():
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = ExampleDocument(text="Mount Everest is the highest peak in the world.", id="doc1")
    doc1.entities.append(LabeledSpan(start=0, end=13, label="LOC"))
    assert str(doc1.entities[0]) == "Mount Everest"

    doc2 = ExampleDocument(text="Alice loves reading books. Bob enjoys playing soccer.", id="doc2")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="PER"))
    assert str(doc2.entities[0]) == "Alice"
    doc2.entities.append(LabeledSpan(start=27, end=30, label="PER"))
    assert str(doc2.entities[1]) == "Bob"
    # we add just one sentence to doc2 that covers only Bob
    doc2.sentences.append(LabeledSpan(start=27, end=53, label="sentence"))
    assert str(doc2.sentences[0]) == "Bob enjoys playing soccer."

    return [doc1, doc2]


def test_taskmodule(unprepared_taskmodule):
    assert unprepared_taskmodule is not None


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents):
    """
    - Prepares the task module with the given documents, i.e. collect available label values.
    - Calls the necessary methods to prepare the task module with the documents.
    - Calls _prepare(documents) and then _post_prepare()

    """
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


def test_prepare(taskmodule):
    assert taskmodule is not None
    assert taskmodule.is_prepared
    assert taskmodule.label_to_id == {"B-LOC": 1, "B-PER": 3, "I-LOC": 2, "I-PER": 4, "O": 0}
    assert taskmodule.id_to_label == {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-PER", 4: "I-PER"}


def test_config(taskmodule):
    config = taskmodule._config()
    assert config["taskmodule_type"] == "MyTokenClassificationTaskModule"
    assert "labels" in config
    assert config["labels"] == ["LOC", "PER"]


@pytest.fixture(scope="module")
def task_encodings_without_targets(taskmodule, documents):
    """
    - Generates task encodings for all the documents, but without associated targets.
    """
    return taskmodule.encode(documents, encode_target=False)


def test_task_encodings_without_targets(task_encodings_without_targets, taskmodule, config):
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs.ids)
        for task_encoding in task_encodings_without_targets
    ]

    # If config is empty
    if config == CONFIG_DEFAULT:
        assert tokens == [
            [
                "[CLS]",
                "mount",
                "everest",
                "is",
                "the",
                "highest",
                "peak",
                "in",
                "the",
                "world",
                ".",
                "[SEP]",
            ],
            [
                "[CLS]",
                "alice",
                "loves",
                "reading",
                "books",
                ".",
                "bob",
                "enjoys",
                "playing",
                "soccer",
                ".",
                "[SEP]",
            ],
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        for t in tokens:
            assert len(t) <= 8

        assert tokens == [
            ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
            ["[CLS]", "highest", "peak", "in", "the", "world", ".", "[SEP]"],
            ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
            ["[CLS]", ".", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
        ]

    # If config has the specified value (max_window=8)
    elif config == CONFIG_MAX_WINDOW:
        for t in tokens:
            assert len(t) <= 8

        assert tokens == [
            ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
            ["[CLS]", "in", "the", "world", ".", "[SEP]"],
            ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
            ["[CLS]", "enjoys", "playing", "soccer", ".", "[SEP]"],
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == CONFIG_PARTITIONS:
        assert tokens

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents):
    return taskmodule.encode(documents, encode_target=True)


def test_task_encodings(task_encodings, taskmodule, config):
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs.ids)
        for task_encoding in task_encodings
    ]
    labels_tokens = [
        [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in task_encoding.targets]
        for task_encoding in task_encodings
    ]
    assert len(labels_tokens) == len(tokens)

    tokens_with_labels = list(zip(tokens, labels_tokens))

    for tokens, labels in tokens_with_labels:
        assert len(tokens) == len(labels)

    # If config is empty
    if config == CONFIG_DEFAULT:
        assert tokens_with_labels == [
            (
                [
                    "[CLS]",
                    "mount",
                    "everest",
                    "is",
                    "the",
                    "highest",
                    "peak",
                    "in",
                    "the",
                    "world",
                    ".",
                    "[SEP]",
                ],
                ["<pad>", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                [
                    "[CLS]",
                    "alice",
                    "loves",
                    "reading",
                    "books",
                    ".",
                    "bob",
                    "enjoys",
                    "playing",
                    "soccer",
                    ".",
                    "[SEP]",
                ],
                ["<pad>", "B-PER", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        for tokens, labels in tokens_with_labels:
            assert len(tokens) <= 8

        assert tokens_with_labels == [
            (
                ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
                ["<pad>", "B-LOC", "I-LOC", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "highest", "peak", "in", "the", "world", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
                ["<pad>", "B-PER", "O", "O", "O", "O", "B-PER", "<pad>"],
            ),
            (
                ["[CLS]", ".", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "B-PER", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified value (max_window=8)
    elif config == CONFIG_MAX_WINDOW:
        for tokens, labels in tokens_with_labels:
            assert len(tokens) <= 8

        assert tokens_with_labels == [
            (
                ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
                ["<pad>", "B-LOC", "I-LOC", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "in", "the", "world", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
                ["<pad>", "B-PER", "O", "O", "O", "O", "B-PER", "<pad>"],
            ),
            (
                ["[CLS]", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == CONFIG_PARTITIONS:
        assert tokens_with_labels == [
            (
                ["[CLS]", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "B-PER", "O", "O", "O", "O", "<pad>"],
            )
        ]

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def task_encodings_for_batch(task_encodings, config):
    # just take everything we have
    return task_encodings


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings_for_batch, config):
    return taskmodule.collate(task_encodings_for_batch)


def test_collate(batch, config):
    assert batch is not None
    assert len(batch) == 2
    inputs, targets = batch

    assert set(inputs.data) == {"input_ids", "attention_mask"}
    input_ids_list = inputs.input_ids.tolist()
    attention_mask_list = inputs.attention_mask.tolist()
    targets_list = targets.tolist()

    # If config is empty
    if config == CONFIG_DEFAULT:
        assert input_ids_list == [
            [101, 4057, 23914, 2003, 1996, 3284, 4672, 1999, 1996, 2088, 1012, 102],
            [101, 5650, 7459, 3752, 2808, 1012, 3960, 15646, 2652, 4715, 1012, 102],
        ]
        assert attention_mask_list == [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        assert targets_list == [
            [-100, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, -100],
            [-100, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, -100],
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        assert input_ids_list == [
            [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
            [101, 3284, 4672, 1999, 1996, 2088, 1012, 102],
            [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            [101, 1012, 3960, 15646, 2652, 4715, 1012, 102],
        ]
        assert attention_mask_list == [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        assert targets_list == [
            [-100, 1, 2, 0, 0, 0, 0, -100],
            [-100, 0, 0, 0, 0, 0, 0, -100],
            [-100, 3, 0, 0, 0, 0, 3, -100],
            [-100, 0, 3, 0, 0, 0, 0, -100],
        ]

    # If config has the specified values (max_window=8)
    elif config == CONFIG_MAX_WINDOW:
        assert input_ids_list == [
            [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
            [101, 1999, 1996, 2088, 1012, 102, 0, 0],
            [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            [101, 15646, 2652, 4715, 1012, 102, 0, 0],
        ]
        assert attention_mask_list == [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]
        assert targets_list == [
            [-100, 1, 2, 0, 0, 0, 0, -100],
            [-100, 0, 0, 0, 0, -100, -100, -100],
            [-100, 3, 0, 0, 0, 0, 3, -100],
            [-100, 0, 0, 0, 0, -100, -100, -100],
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == CONFIG_PARTITIONS:
        assert input_ids_list == [[101, 3960, 15646, 2652, 4715, 1012, 102]]
        assert attention_mask_list == [[1, 1, 1, 1, 1, 1, 1]]
        assert targets_list == [[-100, 3, 0, 0, 0, 0, -100]]

    else:
        raise ValueError(f"unknown config: {config}")

    inputs_expected = BatchEncoding(
        data={
            "input_ids": torch.tensor(input_ids_list, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.int64),
        }
    )
    assert set(inputs.data) == set(inputs_expected.data)
    targets_expected = torch.tensor(targets_list, dtype=torch.int64)
    assert torch.equal(targets, targets_expected)


# This is not used, but can be used to create a batch of task encodings with targets for the unbatched_outputs fixture.
@pytest.fixture(scope="module")
def real_model_output(batch, taskmodule):
    from pytorch_ie.models import TransformerTokenClassificationModel

    model = TransformerTokenClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=len(taskmodule.label_to_id),
    )
    inputs, targets = batch
    result = model(inputs)
    return result


@pytest.fixture(scope="module")
def model_output(config, batch, taskmodule):
    # create "perfect" output from targets
    targets = batch[1].clone()
    targets[targets == -100] = 0
    one_hot_targets = (
        torch.nn.functional.one_hot(targets, num_classes=len(taskmodule.label_to_id)).float()
        * 0.99
        + 0.005
    )
    # convert to logits (logit = log(p/(1-p)))
    logits = torch.log(one_hot_targets / (1 - one_hot_targets))
    return {"logits": logits}


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule, model_output):
    return taskmodule.unbatch_output(model_output)


def test_unbatched_output(unbatched_outputs, config):
    assert unbatched_outputs is not None

    result = [
        {
            "tags": unbatched_output["tags"],
            "probabilities": unbatched_output["probabilities"].round(3).tolist(),
        }
        for unbatched_output in unbatched_outputs
    ]

    # Based on the config, perform assertions for each unbatched output
    if config == CONFIG_DEFAULT:
        assert result == [
            {
                "tags": ["O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "B-PER", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
        ]

    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        assert result == [
            {
                "tags": ["O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "O", "O", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "B-PER", "O", "O", "O", "O", "B-PER", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "O", "B-PER", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
        ]

    elif config == CONFIG_MAX_WINDOW:
        assert result == [
            {
                "tags": ["O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "O", "O", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "B-PER", "O", "O", "O", "O", "B-PER", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
            {
                "tags": ["O", "O", "O", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            },
        ]

    elif config == CONFIG_PARTITIONS:
        assert result == [
            {
                "tags": ["O", "B-PER", "O", "O", "O", "O", "O"],
                "probabilities": [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            }
        ]

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encodings_for_batch, unbatched_outputs, config):
    named_annotations_per_document = defaultdict(list)
    for task_encoding, task_output in zip(task_encodings_for_batch, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
        named_annotations_per_document[task_encoding.document.id].extend(list(annotations))
    return named_annotations_per_document


def test_annotations_from_output(annotations_from_output, config, documents):
    assert annotations_from_output is not None
    # Sort the annotations in each document by start and end positions
    annotations_from_output = {
        doc_id: sorted(annotations, key=lambda x: (x[0], x[1].start, x[1].end))
        for doc_id, annotations in annotations_from_output.items()
    }
    documents_by_id = {doc.id: doc for doc in documents}
    documents_with_annotations = []
    resolved_annotations = defaultdict(list)
    # Check that the number of annotations is correct
    for doc_id, layer_names_and_annotations in annotations_from_output.items():
        new_doc = documents_by_id[doc_id].copy()
        for layer_name, annotation in layer_names_and_annotations:
            assert layer_name == "entities"
            assert isinstance(annotation, LabeledSpan)
            new_doc.entities.predictions.append(annotation)
            resolved_annotations[doc_id].append(str(annotation))
        documents_with_annotations.append(new_doc)

    resolved_annotations = dict(resolved_annotations)
    # Check based on the config
    if config == CONFIG_DEFAULT:
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice", "Bob"]}

    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        # We get two annotations for Bob because the window overlaps with the previous one.
        # This is not a problem because annotations get de-duplicated during serialization.
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice", "Bob", "Bob"]}

    elif config == CONFIG_MAX_WINDOW:
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice", "Bob"]}

    elif config == CONFIG_PARTITIONS:
        assert resolved_annotations == {"doc2": ["Bob"]}

    else:
        raise ValueError(f"unknown config: {config}")


def test_document_type():
    taskmodule = MyTokenClassificationTaskModule(tokenizer_name_or_path="bert-base-uncased")
    assert taskmodule.document_type == TextDocumentWithLabeledSpans


def test_document_type_with_partitions():
    taskmodule = MyTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", partition_annotation="labeled_partitions"
    )
    assert taskmodule.document_type == TextDocumentWithLabeledSpansAndLabeledPartitions


def test_document_type_with_non_default_span_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = MyTokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased", span_annotation="entities"
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "span_annotation=entities is not the default value ('labeled_spans'), so the taskmodule "
        "MyTokenClassificationTaskModule can not request the usual document type "
        "(TextDocumentWithLabeledSpans) for auto-conversion because this has the bespoken default value "
        "as layer name(s) instead of the provided one(s)."
    )


def test_document_type_with_non_default_partition_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = MyTokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased", partition_annotation="sentences"
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "partition_annotation=sentences is not the default value ('labeled_partitions'), "
        "so the taskmodule MyTokenClassificationTaskModule can not request the usual document type "
        "(TextDocumentWithLabeledSpansAndLabeledPartitions) for auto-conversion because this has "
        "the bespoken default value as layer name(s) instead of the provided one(s)."
    )


def test_document_type_with_non_default_span_and_partition_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = MyTokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased",
            span_annotation="entities",
            partition_annotation="sentences",
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "span_annotation=entities is not the default value ('labeled_spans') and "
        "partition_annotation=sentences is not the default value ('labeled_partitions'), "
        "so the taskmodule MyTokenClassificationTaskModule can not request the usual document "
        "type (TextDocumentWithLabeledSpansAndLabeledPartitions) for auto-conversion because "
        "this has the bespoken default value as layer name(s) instead of the provided one(s)."
    )
