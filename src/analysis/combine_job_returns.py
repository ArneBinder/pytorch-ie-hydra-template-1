import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=False,
)
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import MultiIndex

from src.analysis.common import read_nested_jsons

logger = logging.getLogger(__name__)


def separate_path_and_id(
    path_and_maybe_id: str, separator: str = ":"
) -> Tuple[Optional[str], str]:
    parts = path_and_maybe_id.split(separator, 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def get_file_paths(paths_file: str, file_name: str, use_aggregated: bool) -> Dict[str, str]:
    with open(paths_file, "r") as f:
        paths_maybe_with_ids = f.readlines()
    ids, paths = zip(*[separate_path_and_id(path.strip()) for path in paths_maybe_with_ids])

    if use_aggregated:
        file_base_name, ext = os.path.splitext(file_name)
        file_name = f"{file_base_name}.aggregated{ext}"
    file_paths = [os.path.join(path, file_name) for path in paths]
    return {
        id if id is not None else f"idx={idx}": path
        for idx, (id, path) in enumerate(zip(ids, file_paths))
    }


def get_job_id_col(index: pd.MultiIndex) -> Optional[Tuple]:
    for idx in index:
        if "job_id" in idx:
            return idx
    return None


def remove_part_from_multi_index(index: pd.MultiIndex, part: str) -> pd.MultiIndex:
    new_index = []
    for idx in index:
        new_idx = tuple([i for i in idx if i != part])
        new_index.append(new_idx)
    return MultiIndex.from_tuples(new_index)


def main(
    paths_file: str,
    file_name: str,
    use_aggregated: bool,
    columns: Optional[List[str]],
    round_precision: Optional[int],
    format: str,
    transpose: bool = False,
    unpack_multirun_results: bool = False,
    unpack_multirun_results_with_job_id: bool = False,
    in_percent: bool = False,
    reset_index: bool = False,
):
    file_paths = get_file_paths(
        paths_file=paths_file, file_name=file_name, use_aggregated=use_aggregated
    )
    data = read_nested_jsons(json_paths=file_paths)

    job_id_col = get_job_id_col(data.columns)

    if columns is not None:
        columns_multi_index = [
            tuple([part or np.nan for part in col.split("/")]) for col in columns
        ]
        if unpack_multirun_results_with_job_id:
            if job_id_col is None:
                raise ValueError("Job ID column not found in the data.")
            if job_id_col not in columns_multi_index:
                columns_multi_index.append(job_id_col)
        try:
            available_cols = data.columns.tolist()
            for col in columns_multi_index:
                if col not in available_cols:
                    raise KeyError(f"Column {col} not found in the data.")
            data_series = [data[col] for col in columns_multi_index]
        except KeyError as e:
            print(
                f"Columns {columns_multi_index} not found in the data. Available columns are {list(data.columns)}."
            )
            raise e
        data = pd.concat(data_series, axis=1)

    # drop rows that are all NaN
    data = data.dropna(how="all")

    # if more than one data point, drop the index levels that are everywhere the same
    if len(data) > 1:
        unique_levels = [
            idx
            for idx, level in enumerate(data.index.levels)
            if len(data.index.get_level_values(idx).unique()) == 1
        ]
        for level in sorted(unique_levels, reverse=True):
            data.index = data.index.droplevel(level)

    # if more than one column, drop the columns that are everywhere the same
    if len(data.columns) > 1:
        unique_column_levels = [
            idx
            for idx, level in enumerate(data.columns.levels)
            if len(data.columns.get_level_values(idx).unique()) == 1
        ]
        for level in sorted(unique_column_levels, reverse=True):
            data.columns = data.columns.droplevel(level)

    if unpack_multirun_results or unpack_multirun_results_with_job_id:
        index_names = list(data.index.names)
        data_series_lists = data.copy()
        job_ids = None
        if job_id_col in data_series_lists.columns:
            job_ids_series = data_series_lists.pop(job_id_col)
            job_ids_frame = pd.DataFrame(pd.DataFrame.from_records(job_ids_series.values))
            job_ids_frame.index = job_ids_series.index
            # check that all rows are identical
            if job_ids_frame.nunique().max():
                job_ids = job_ids_frame.iloc[0]
            else:
                logger.warning(
                    "Job IDs are not identical across all rows. Cannot unpack "
                    "multirun results with job ids as columns."
                )

        while not isinstance(data_series_lists, pd.Series):
            data_series_lists = data_series_lists.stack(future_stack=True)
        data_series_lists = data_series_lists.dropna()
        data = pd.DataFrame.from_records(data_series_lists.values, index=data_series_lists.index)
        if job_ids is not None:
            data.columns = job_ids
        num_col_levels = data.index.nlevels - len(index_names)
        for _ in range(num_col_levels):
            data = data.unstack()
        data.columns = data.columns.swaplevel(0, -1)
        data = data.dropna(how="all", axis="columns")

    # needs to happen before rounding, otherwise the rounding will be off
    if in_percent:
        data = data * 100

    if round_precision is not None:
        data = data.round(round_precision)

    # needs to happen before transposing
    if format == "markdown_mean_and_std":
        if data.columns.nlevels == 1:
            data.columns = pd.MultiIndex.from_tuples([(col,) for col in data.columns.tolist()])

        # get mean columns
        mean_col_names = [col for col in data.columns if "mean" in col]
        mean_columns = data[mean_col_names].copy()
        # remove all "mean" from col names
        mean_columns.columns = remove_part_from_multi_index(mean_columns.columns, "mean")
        # get std columns
        std_col_names = [col for col in data.columns if "std" in col]
        std_columns = data[std_col_names].copy()
        # remove all "std" from col names
        std_columns.columns = remove_part_from_multi_index(std_columns.columns, "std")
        # sanity check
        if not mean_columns.columns.equals(std_columns.columns):
            raise ValueError("Mean and std columns do not match.")
        mean_and_std = mean_columns.astype(str) + " ± " + std_columns.astype(str)
        mean_and_std.columns = [
            ("mean ± std",) + (tuple(col) if col != ((),) else ()) for col in mean_columns.columns
        ]
        # remove mean and std columns from data
        # we can not use drop because the columns is a multiindex that may contain NaNs
        other_cols = [
            col for col in data.columns if col not in set(mean_col_names + std_col_names)
        ]
        data = data[other_cols]
        # add mean and std columns to data
        data = pd.concat([data, mean_and_std], axis=1)
        if data.columns.nlevels == 1:
            data.columns = data.columns.to_flat_index()
            data.columns = [
                "/".join(col) if isinstance(col, tuple) else col for col in data.columns
            ]

    if transpose:
        data = data.T

    if reset_index:
        data = data.reset_index()

    if format in ["markdown", "markdown_mean_and_std"]:
        print(data.to_markdown(index=not reset_index))
    elif format == "json":
        print(data.to_json())
    else:
        raise ValueError(f"Invalid format: {format}. Use 'markdown' or 'json'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine job returns and show as Markdown table or Json"
    )
    parser.add_argument(
        "--paths-file", type=str, help="Path to the file containing the paths to the job returns"
    )
    parser.add_argument(
        "--use-aggregated", action="store_true", help="Whether to use the aggregated job returns"
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="job_return_value.json",
        help="Name of the file to write the aggregated job returns to",
    )
    parser.add_argument(
        "--columns", type=str, nargs="+", help="Columns to select from the combined job returns"
    )
    parser.add_argument(
        "--unpack-multirun-results", action="store_true", help="Unpack multirun results"
    )
    parser.add_argument(
        "--unpack-multirun-results-with-job-id",
        action="store_true",
        help="Unpack multirun results with job ID",
    )
    parser.add_argument("--transpose", action="store_true", help="Transpose the table")
    parser.add_argument(
        "--round-precision",
        type=int,
        help="Round the values in the combined job returns to the specified precision",
    )
    parser.add_argument(
        "--in-percent", action="store_true", help="Show the values in percent (multiply by 100)"
    )
    parser.add_argument(
        "--reset-index", action="store_true", help="Reset the index of the combined job returns"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "markdown_mean_and_std", "json"],
        help="Format to output the combined job returns",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
