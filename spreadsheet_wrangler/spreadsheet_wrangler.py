"""
spreadsheet_wrangler.py
"""

import ast
import copy
import json
import logging
import re

import numpy as np
import pandas as pd

log_ = logging.getLogger("spreadsheet-wrangler")
# from .file_io import *


def read_pseodonyms(string: str) -> dict:
    if len(string.strip()) == 0:
        return {}
    return json.loads(string)


def make_unique(df: pd.DataFrame, column: str, prefer_column=None):
    """Remove the values that are duplicates, prefer the rows with a value.
    column is the unique value column, prefer_column is the one to look at the longest argument of

    column: column to remove duplicates
    perfer_column: If there are duplicate rows then the row with the
    longest value of this value is kept
    """

    not_unique_values = [val for val in df[column] if list(df[column]).count(val) > 1]
    for value in not_unique_values:
        while list(df[column]).count(value) > 1:
            drop_rows = []
            if prefer_column is None:
                for i, _ in df[df[column] == value].iterrows():
                    drop_rows.append(i)
                drop_rows.pop()  # keep last row
            else:
                matching = df[df[column] == value][prefer_column]
                min_length = min([str(pt) for pt in matching])
                for i, row in df[df[column] == value].iterrows():
                    if len(str(row[prefer_column])) == min_length:
                        drop_rows.append(i)
            df = df.drop(df.index[drop_rows])
    return df


def extract_columns_by_pseudonyms(df: pd.DataFrame, column_names: dict) -> pd.DataFrame:
    """Finds knowns pseudonyms for columns and includes names them correctly for passing as argument"""
    for name in df.columns:
        for column, names in column_names.items():
            if (
                name.lower() in [pt.lower() for pt in names]
                or name.lower() == column.lower()
            ):
                df = df.rename(columns={name: column})
    return df


def uncluster_ast(df: pd.DataFrame, grouped_column: str) -> pd.DataFrame:
    formated_df = df[df[grouped_column] != np.nan]
    expanded_rows = []
    for _, row in formated_df.iterrows():
        group = ast.literal_eval(row[grouped_column])
        for item in group:
            row[grouped_column] = item
            expanded_rows.append(copy.deepcopy(row))
    return pd.DataFrame(expanded_rows)


def uncluster_regex(
    df: pd.DataFrame, grouped_column: str, expression: str = "[A-z]+[0-9]+"
) -> pd.DataFrame:
    formated_df = df[df[grouped_column] != np.nan]
    expanded_rows = []
    ref_regex = re.compile(expression)
    for _, row in formated_df.iterrows():
        refs = row[grouped_column]
        if not isinstance(refs, str):
            continue
        for ref in ref_regex.findall(refs):
            row[grouped_column] = ref
            expanded_rows.append(copy.deepcopy(row))
    return pd.DataFrame(expanded_rows)


def uncluster(df: pd.DataFrame, grouped_column: str) -> pd.DataFrame:
    return uncluster_regex(df, grouped_column)


def cluster(df: pd.DataFrame, on: list, column: str) -> pd.DataFrame:
    """ref-des will not be a tuple of all the matching lines, the rest of the line is taken to be the first in the file and carried forward"""
    for pt in on:
        if pt not in df.columns:
            msg = f"column {pt} or pseudonym not found"
            raise KeyError(msg)
    if column not in df.columns:
        msg = f"column {column} or pseudonym not found"
        raise KeyError(msg)

    grouped = df.groupby(
        by=list(on)
    )  #  IMPORTANT: This has to be a list as a tuple is interpreted as a single key.
    clustered: list = []
    rows: list = []

    for _, group in grouped:
        cluster_entries = []
        for _i, row in group.iterrows():
            cluster_entries.append(row[column])
        copy_row = copy.deepcopy(row)
        rows.append(copy_row)
        clustered.append(tuple(cluster_entries))

    df = pd.DataFrame(rows)
    index = df.columns.get_indexer_for([column])[0]
    df = df.drop(columns=[column])
    df.insert(int(index), column=column, value=clustered)
    return df


def compare(left: pd.DataFrame, right: pd.DataFrame, columns: str, on: str) -> dict:
    errors: dict = {"line": [], "column": [], "description": []}
    for pt in list(left[on]) + list(right[on]):
        matching_rows_left = left[left[on] == pt]
        matching_rows_right = right[right[on] == pt]
        for column in columns:
            for lc, rc in zip(
                matching_rows_left[column], matching_rows_right[column], strict=False
            ):
                if lc != rc:
                    # filter out nan
                    if lc is np.nan and rc is np.nan:
                        continue
                    errors["line"].append(pt)
                    errors["column"].append(column)
                    errors["description"].append(
                        f"{lc} ({type(lc)}) != {rc} ({type(rc)})"
                    )
    return errors


def select_on_value(
    df: pd.DataFrame, value, column: str, *, blank_defaults: bool = True
):
    """
    Selects all rows that match the value in the given column
    """
    match = [str(pt).lower() == str(value).lower() for pt in df[column]]
    return df.loc[match | df[column].isna()] if blank_defaults else df.loc[match]


# def get_unique(df: pd.DataFrame, column: str, blank_defaults: bool) -> pd.DataFrame:
#    """
#    Selects only unique lines matching the column, value
#    """
#    return make_unique(df, column=column, prefer_column=on)


def filter_df(
    df: pd.DataFrame, on: str, value, column: str, *, blank_defaults: bool = True
) -> pd.DataFrame:
    """
    Returns the rows that both match
    """
