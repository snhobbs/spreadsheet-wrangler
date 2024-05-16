"""
spreadsheet_wrangler.py
"""

import ast
import copy
import csv
import json
import re
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# from pandas_ods_reader import read_ods
import pyexcel_ods3


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


def read_csv_to_df(fname: str, **kwargs) -> pd.DataFrame:
    # Use automatic dialect detection by setting sep to None and engine to python
    # try sniffing
    try:
        # Use automatic dialect detection by setting sep to None and engine to python
        kwargs["sep"] = None
        kwargs["delimiter"] = None
        return pd.read_csv(fname, engine="python", **kwargs)
    except Exception:
        pass
    try:
        kwargs["sep"] = ","
        return pd.read_csv(fname, **kwargs)
    except Exception:
        pass
    try:
        kwargs["sep"] = ";"
        return pd.read_csv(fname, **kwargs)
    except Exception:
        raise


def read_ods_format_to_df(fname: str, **kwargs) -> pd.DataFrame:
    data = pyexcel_ods3.get_data(fname, **kwargs)
    ave_line_length = np.mean([len(line) for line in data])
    data_lines = [
        line for line in data if len(line) >= ave_line_length
    ]  # assume this is the data
    header = data_lines[0]
    data_lines = data_lines[1:]
    df_dict = {column: [] for column in header}
    for line in data_lines:
        for column, pt in zip(df_dict.keys(), line, strict=False):
            df_dict[column].append(pt)
    return pd.DataFrame(df_dict)


def get_supported_file_types_df():
    """
    Installed readers
    """
    return [
        {
            "title": "text",
            "kwargs": {
                "header": 0,
                "skipinitialspace": True,
                "index_col": None,
                "comment": "#",
                "quotechar": '"',
                "quoting": csv.QUOTE_MINIMAL,
                "engine": "python",
                "skip_blank_lines": True,
            },
            "extensions": ("csv", "txt"),
            "writedf": pd.DataFrame.to_csv,
            "readf": read_csv_to_df,
        },
        {
            "title": "excel",
            "kwargs": {"sheet_name": 0, "header": 0, "skiprows": 0},
            "extensions": ("xls", "xlsx", "xlsm", "xlsb"),
            "writedf": pd.DataFrame.to_excel,
            "readf": pd.read_excel,
        },
        {
            "title": "ods",
            "kwargs": {"sheet_name": 0, "header": 0, "skiprows": 0},
            "extensions": ("ods", "odt", "odf"),
            "writedf": None,
            "readf": read_ods_format_to_df,
        },
    ]


def get_supported_file_formats() -> tuple[str]:
    """
    returns collection of all the supported extensions
    """
    extensions = []
    for entry in get_supported_file_types_df():
        extensions.extend(entry["extensions"])
    return tuple(extensions)


def write(df: pd.DataFrame, fname: str, **kwargs) -> None:
    ext = Path(fname).suffix.strip(".").lower()
    types = get_supported_file_types_df()
    writer = None
    for value in types:
        if ext in value["extensions"]:
            writer = value["writedf"]
            break

    assert writer
    writer(df, fname, **kwargs)


def read_file_to_df(fname: str, **kwargs) -> pd.DataFrame:
    """
    Cycle through extensions, use the reader object to call
    """
    ext = Path(fname).suffix.strip(".").lower()
    df = None
    found = False
    for reader in get_supported_file_types_df():
        if ext in reader["extensions"]:
            found = True
            if kwargs is None:
                kwargs = reader["kwargs"]
            df = reader["readf"](fname, **kwargs)
            break
    if not found:
        msg = f"Extension {ext} unsupported"
        raise UserWarning(msg)
    return pd.DataFrame(df)


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


def select_on_value(df: pd.DataFrame, value, column: str, blank_defaults: bool):
    """
    Selects all rows that match the value in the given column
    """
    match = [str(pt).lower() == str(value).lower() for pt in df[column]]
    if blank_defaults:
        filtered_df = df.loc[(match) | (df[column].isnull())]
    else:
        filtered_df = df.loc[(match)]
    return filtered_df


# def get_unique(df: pd.DataFrame, column: str, blank_defaults: bool) -> pd.DataFrame:
#    """
#    Selects only unique lines matching the column, value
#    """
#    return make_unique(df, column=column, prefer_column=on)


def filter_df(
    df: pd.DataFrame, on: str, value, column: str, blank_defaults: bool
) -> pd.DataFrame:
    """
    Returns the rows that both match
    """
