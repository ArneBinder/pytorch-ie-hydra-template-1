# PIE Dataset Card for "DrugProt"

This is a [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie) wrapper for the
[DrugProt Huggingface dataset loading script](https://huggingface.co/datasets/bigbio/drugprot).

## Data Schema

There are two versions of the dataset supported, `drugprot_source` and `drugprot_bigbio_kb`.

#### `DrugprotDocument` for `drugprot_source`

defines following fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)
- `title` (str, optional)
- `abstract` (str, optional)

and the following annotation layers:

- `entities` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `entities`)

#### `DrugprotBigbioDocument` for `drugprot_bigbio_kb`

defines following fields:

- `text` (str)
- `id` (str, optional)
- `metadata` (dictionary, optional)

and the following annotation layers:

- `passages` (annotation type: `LabeledSpan`, target: `text`)
- `entities` (annotation type: `LabeledSpan`, target: `text`)
- `relations` (annotation type: `BinaryRelation`, target: `entities`)

See [here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/annotations.py) for the annotation
type definitions.

## Document Converters

The dataset provides predefined document converters for the following target document types:

- `pie_documents.documents.TextDocumentWithLabeledSpansAndBinaryRelations` for `DrugprotDocument`
- `pie_documents.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions` for `DrugprotBigbioDocument`

See [here](https://github.com/ArneBinder/pie-documents/blob/main/src/pie_documents/documents.py) for the document type
definitions.
