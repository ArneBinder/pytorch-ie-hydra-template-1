import json
from typing import Dict, List, Optional

import pandas as pd


def parse_identifier(
    identifier_str, defaults: Dict[str, str], parts_sep: str = ",", key_val_sep: str = "="
) -> Dict[str, str]:
    parts = [
        part.split(key_val_sep)
        for part in identifier_str.strip().split(parts_sep)
        if key_val_sep in part
    ]
    parts_dict = dict(parts)
    return {**defaults, **parts_dict}


def read_nested_json(path: str) -> pd.DataFrame:
    # Read the nested JSON data into a pandas DataFrame
    with open(path, "r") as f:
        data = json.load(f)
    result = pd.json_normalize(data, sep="/")
    result.index.name = "entry"
    return result


def read_nested_jsons(
    json_paths: Dict[str, str],
    default_key_values: Optional[Dict[str, str]] = None,
    column_level_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    identifier_strings = json_paths.keys()
    dfs = [read_nested_json(json_paths[identifier_str]) for identifier_str in identifier_strings]
    new_index_levels = pd.MultiIndex.from_frame(
        pd.DataFrame(
            [
                parse_identifier(identifier_str, default_key_values or {})
                for identifier_str in identifier_strings
            ]
        )
    )
    dfs_concat = pd.concat(dfs, keys=list(new_index_levels), names=new_index_levels.names, axis=0)
    dfs_concat.columns = pd.MultiIndex.from_tuples(
        [col.split("/") for col in dfs_concat.columns], names=column_level_names
    )
    return dfs_concat
