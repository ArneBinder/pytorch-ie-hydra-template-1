import glob
import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Type, TypeVar

from pie_datasets.core.dataset_dict import METADATA_FILE_NAME
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Document
from pytorch_ie.utils.hydra import resolve_optional_document_type, serialize_document_type

from src.serializer.interface import DocumentSerializer
from src.utils import get_pylogger

log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


def get_era(ann_file_path):
    entities = []
    relations = []
    attributes = []
    with open(ann_file_path, "r") as file:
        for line in file:
            if line.startswith("T"):
                parts = line.split("\t")

                if ";" in parts[1]:
                    label_ent_1, ent_2 = parts[1].split(";")
                    label, start_idx_1, end_idx_1 = label_ent_1.split()
                    start_idx_2, end_idx_2 = ent_2.split()
                    start_idx_1, end_idx_1 = map(int, (start_idx_1, end_idx_1))
                    start_idx_2, end_idx_2 = map(int, (start_idx_2, end_idx_2))
                    text_1, text_2 = (
                        parts[2][: end_idx_1 - start_idx_1],
                        parts[2][end_idx_1 - start_idx_1 :],
                    )
                    entities.append(
                        {
                            "id": parts[0],
                            "label": label,
                            "start_idx_1": start_idx_1,
                            "end_idx_1": end_idx_1,
                            "start_idx_2": start_idx_2,
                            "end_idx_2": end_idx_2,
                            "text_1": text_1.strip(),
                            "text_2": text_2.strip(),
                            "len_1": end_idx_1 - start_idx_1,
                            "len_2": end_idx_2 - start_idx_2,
                            "multispan": True,
                        }
                    )
                else:
                    text = parts[2].strip()
                    label, start_idx, end_idx = parts[1].split()
                    start_idx, end_idx = map(int, (start_idx, end_idx))

                    entities.append(
                        {
                            "id": parts[0],
                            "label": label,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "text": text,
                            "len": end_idx - start_idx,
                            "multispan": False,
                        }
                    )
            elif line.startswith("R"):
                parts = line.split("\t")
                label, arg1, arg2 = parts[1].split()
                arg1 = arg1.split(":")[-1]
                arg2 = arg2.split(":")[-1]
                relations.append(
                    {
                        "label": label,
                        "arg1": arg1,
                        "arg2": arg2,
                    }
                )
            elif line.startswith("A"):
                parts = line.split("\t")[1].split()
                if len(parts) == 2:

                    attributes.append(
                        {
                            "label": parts[0],
                            "entity": parts[1],
                        }
                    )
                elif len(parts) == 3:
                    label, entity, value = line.split("\t")[1].split()
                    attributes.append(
                        {
                            "label": label,
                            "entity": entity,
                            "value": value,
                        }
                    )
    return entities, relations, attributes


class BratSerializer(DocumentSerializer):
    def __init__(self, **kwargs):
        self.default_kwargs = kwargs

    def __call__(self, documents: Sequence[Document], **kwargs) -> Dict[str, str]:
        return self.write_with_defaults(documents=documents, **kwargs)

    def write_with_defaults(self, **kwargs) -> Dict[str, str]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.write(**all_kwargs)

    @classmethod
    def write(
        cls,
        documents: Sequence[Document],
        path: str,
        metadata_file_name: str = METADATA_FILE_NAME,
        split: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        realpath = os.path.realpath(path)
        log.info(f'serialize documents to "{realpath}" ...')
        os.makedirs(realpath, exist_ok=True)

        if len(documents) == 0:
            raise Exception("cannot serialize empty list of documents")
        document_type = type(documents[0])
        metadata = {"document_type": serialize_document_type(document_type)}
        full_metadata_file_name = os.path.join(realpath, metadata_file_name)

        if split is not None:
            realpath = os.path.join(realpath, split)
            os.makedirs(realpath, exist_ok=True)
        metadata_text = defaultdict(str)
        for doc in documents:
            file_name = f"{doc.id}.ann"
            metadata_text[f"{file_name}"] = doc.text
            ann_path = os.path.join(realpath, file_name)
            with open(ann_path, "w+") as f:
                entities = doc.entities.predictions
                relations = doc.relations.predictions
                entities_to_merge = [
                    (rel.head, rel.tail) for rel in relations if rel.label == "parts_of_same"
                ]
                j = 0
                span2id: DefaultDict[LabeledSpan, str] = defaultdict()
                for entity1, entity2 in entities_to_merge:

                    entity_id = f"T{j}"

                    start_idx_1 = entity1.start
                    end_idx_1 = entity1.end
                    label1 = entity1.label
                    entity_text_1 = doc.text[start_idx_1:end_idx_1]

                    start_idx_2 = entity2.start
                    end_idx_2 = entity2.end
                    label2 = entity2.label
                    entity_text_2 = doc.text[start_idx_2:end_idx_2]

                    if label1 != label2:
                        raise Exception(
                            "Entities with parts_of_same relation should have same entity type."
                        )

                    entry = f"{entity_id}\t{label1} {start_idx_1} {end_idx_1};{start_idx_2} {end_idx_2}\t{entity_text_1} {entity_text_2}\n"
                    span2id[entity1] = entity_id
                    span2id[entity2] = entity_id
                    f.write(entry)

                    entities.remove(entity1)
                    entities.remove(entity2)
                    j += 1

                for i, entity in enumerate(entities, start=j):
                    entity_id = f"T{i}"
                    start_idx = entity.start
                    end_idx = entity.end
                    label = entity.label
                    entity_text = doc.text[start_idx:end_idx]
                    entry = f"{entity_id}\t{label} {start_idx} {end_idx}\t{entity_text}\n"
                    span2id[entity] = entity_id
                    f.write(entry)

                for i, relation in enumerate(relations):
                    relation_id = f"R{i}"
                    arg1 = span2id[relation.head]
                    arg2 = span2id[relation.tail]
                    label = relation.label
                    if label == "parts_of_same":
                        continue
                    entry = f"{relation_id}\t{label} Arg1:{arg1} Arg2:{arg2}\n"
                    f.write(entry)

        metadata["text"] = metadata_text
        print(realpath)

        if os.path.exists(full_metadata_file_name):
            log.warning(
                f"metadata file {full_metadata_file_name} already exists, "
                "it will be overwritten!"
            )
        with open(full_metadata_file_name, "w") as f:
            json.dump(metadata, f, indent=2)
        return {"path": realpath, "metadata_file_name": metadata_file_name}

    @classmethod
    def read(
        cls,
        path: str,
        document_type: Optional[Type[D]] = None,
        metadata_file_name: str = METADATA_FILE_NAME,
        split: Optional[str] = None,
    ) -> List[D]:
        realpath = os.path.realpath(path)
        log.info(f'load documents from "{realpath}" ...')

        # try to load metadata including the document_type
        full_metadata_file_name = os.path.join(realpath, metadata_file_name)
        if os.path.exists(full_metadata_file_name):
            with open(full_metadata_file_name) as f:
                metadata = json.load(f)
            document_type = resolve_optional_document_type(metadata.get("document_type"))

        if document_type is None:
            raise Exception("document_type is required to load serialized documents")

        if split is not None:
            realpath = os.path.join(realpath, split)
        ann_files = glob.glob(os.path.join(realpath, "*.ann"))
        file_name2text = metadata.get("text")
        documents = []
        for ann_file in ann_files:
            _, file_name = os.path.split(ann_file)
            doc_text = file_name2text[file_name]
            entities, relations, attributes = get_era(ann_file)
            document = document_type(text=doc_text, id=file_name.split(".")[0])
            idx2entity: DefaultDict[str, LabeledSpan] = defaultdict()
            candidate_ents = []

            for entity in entities:
                if entity["multispan"]:
                    idx = entity["id"]
                    label = entity["label"]
                    text_1 = entity["text_1"]
                    text_2 = entity["text_2"]

                    start_idx_1 = entity["start_idx_1"]
                    end_idx_1 = entity["end_idx_1"]
                    start_idx_2 = entity["start_idx_2"]
                    end_idx_2 = entity["end_idx_2"]
                    ent1 = LabeledSpan(start=start_idx_1, end=end_idx_1, label=label)
                    ent2 = LabeledSpan(start=start_idx_2, end=end_idx_2, label=label)
                    idx2entity[idx] = ent1
                    idx2entity[idx] = ent2
                    if (
                        text_1 != document.text[start_idx_1:end_idx_1]
                        or text_2 != document.text[start_idx_2:end_idx_2]
                    ):
                        raise Exception("TODO:Entity Span mismatch")
                    document.entities.predictions.append(ent1)
                    document.entities.predictions.append(ent2)
                    candidate_ents.append((ent1, ent2))
                else:
                    idx = entity["id"]
                    start = entity["start_idx"]
                    end = entity["end_idx"]
                    label = entity["label"]
                    text = entity["text"]
                    ent = LabeledSpan(start=start, end=end, label=label)
                    idx2entity[idx] = ent
                    if text != document.text[start:end]:
                        raise Exception("TODO:Entity Span mismatch")
                    document.entities.predictions.append(ent)

            for relation in relations:
                head = idx2entity[relation["arg1"]]
                tail = idx2entity[relation["arg2"]]
                label = relation["label"]
                rel = BinaryRelation(head=head, tail=tail, label=label)
                document.relations.predictions.append(rel)

            if candidate_ents:
                for ent1, ent2 in candidate_ents:
                    rel = BinaryRelation(head=ent1, tail=ent2, label="parts_of_same")
                    document.relations.predictions.append(rel)

            documents.append(document)

        return documents

    def read_with_defaults(self, **kwargs) -> List[D]:
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.read(**all_kwargs)
