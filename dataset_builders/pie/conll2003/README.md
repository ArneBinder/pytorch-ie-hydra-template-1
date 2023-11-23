# PIE Dataset Card for "conll2003"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[CoNLL 2003 Huggingface dataset loading script](https://huggingface.co/datasets/conll2003).

## Data Schema

The document type for this dataset is `CoNLL2003Document` which defines the following data fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `entities` (annotation type: `LabeledSpan`, target: `text`)

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/annotations.py) for the annotation type definitions.

## Document Converters

The dataset provides document converters for the following target document types:

- `pytorch_ie.documents.TextDocumentWithLabeledSpans`

See [here](https://github.com/ChristophAlt/pytorch-ie/blob/main/src/pytorch_ie/documents.py) for the document type
definitions.
