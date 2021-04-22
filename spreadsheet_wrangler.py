'''
spreadsheet_wrangler.py
Version 0.1.1
'''
import click
import pandas as pd  # type: ignore
import csv
import os
import numpy as np # type: ignore
import copy
import ast
import json
import re
#from pandas_ods_reader import read_ods
import pyexcel_ods3

supported_text_formats = (".csv", ".txt")
supported_excel_formats = (
    ".xls", ".xlsx", ".xlsm",
    ".xlsb"
)
supported_ods_formats = (".ods", ".odt", ".odf")
supported_file_formats = tuple(list(supported_text_formats) + list(supported_excel_formats) + list(supported_ods_formats))

def read_pseodonyms(string: str) -> dict:
    if len(string.strip()) == 0:
        return {}
    return json.loads(string)

'''Remove the values that are duplicates, prefer the rows with a value. column is the unique value column, prefer_column is the one to look at the longest argument of'''
def make_unique(df: pd.DataFrame, column: str, prefer_column=None):
    not_unique_values = [val for val in df[column] if list(df[column]).count(val) > 1]
    for value in not_unique_values:
        while list(df[column]).count(value) > 1:
            drop_rows = []
            if prefer_column is None:
                for i, row in df[df[column] == value].iterrows():
                    drop_rows.append(i)
                drop_rows.pop() # keep last row
            else:
                min_length = min([str(pt) for pt in df[df[column] == value][prefer_column]])
                for i, row in df[df[column] == value].iterrows():
                    if len(str(row[prefer_column])) == min_length:
                        drop_rows.append(i)
            df.drop(df.index[drop_rows], inplace=True)
    return df

'''Finds knowns pseudonyms for columns and includes names them correctly for passing as argument'''
def extract_columns_by_pseudonyms(df: pd.DataFrame, column_names: dict) -> pd.DataFrame:
    included : list = list()
    for name in df.columns:
        for column, names in column_names.items():
            if name.lower() in [pt.lower() for pt in names] or name.lower() == column.lower():
                df.rename(columns={name:column}, inplace=True)
    return df

def read_csv_to_df(fname: str) -> pd.DataFrame:
    # Use automatic dialect detection by setting sep to None and engine to python
    kwargs = dict(
        header=0, skipinitialspace=True,
        index_col=None, comment="#", quotechar='"',
        quoting=csv.QUOTE_MINIMAL, engine="python", skip_blank_lines=True
    )
    #try sniffing
    try:
        # Use automatic dialect detection by setting sep to None and engine to python
        kwargs["sep"]=None
        kwargs["delimiter"]=None
        df = pd.read_csv(fname, **kwargs)
        return df
    except Exception as e:
        print(e)
        pass
    try:
        kwargs["sep"]=','
        df = pd.read_csv(fname, **kwargs)
        return df
    except Exception as e:
        print(e)
        pass
    try:
        kwargs["sep"]=';'
        df = pd.read_csv(fname, **kwargs)
        return df
    except Exception as e:
        raise

def read_file_to_df(fname: str) -> dict:
    name, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext in supported_text_formats:
        df = read_csv_to_df(fname)

    elif ext in supported_excel_formats :
        df = pd.read_excel(fname, sheet_name=0, header=0, skiprows=0,
                comment="#", skip_blank_lines=True)

    elif ext in supported_ods_formats :
        data = pyexcel_ods3.get_data(fname)
        ave_line_length = np.mean([len(line) for line in data])
        data_lines = []
        for line in data:
            if len(line) >= ave_line_length: # assume this is the data
                data_lines.append(line)
        header = data_lines[0]
        data_lines = data_lines[1:]
        df_dict = dict([(column, []) for column in header])
        for line in data_lines:
            for column, pt in zip(df_dict.keys(), line):
                df_dict[column].append(pt)

        #read_ods(fname, sheet=0)


    return pd.DataFrame(df_dict)

def uncluster_ast(df: pd.DataFrame, grouped_column: str) -> pd.DataFrame:
    formated_df = df[df[grouped_column] != np.nan]
    expanded_rows = []
    for _, row in formated_df.iterrows():
        group = ast.literal_eval(row[grouped_column])
        for item in group:
            row[grouped_column] = item
            expanded_rows.append(copy.deepcopy(row))
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


def uncluster_regex(df: pd.DataFrame, grouped_column: str, expression: str = "[A-z]+[0-9]+") -> pd.DataFrame:
    formated_df = df[df[grouped_column] != np.nan]
    expanded_rows = []
    ref_regex = re.compile(expression)
    for _, row in formated_df.iterrows():
        refs = row[grouped_column]
        if type(refs) != str:
            continue
        for ref in ref_regex.findall(refs):
            row[grouped_column] = ref
            expanded_rows.append(copy.deepcopy(row))
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


def uncluster(df: pd.DataFrame, grouped_column: str) -> pd.DataFrame:
    return uncluster_ast(df, grouped_column)

'''ref-des will not be a tuple of all the matching lines, the rest of the line is taken to be the first in the file and carried forward'''
def cluster(df: pd.DataFrame, on: list, column: str) -> pd.DataFrame:
    for pt in on:
        if pt not in df.columns:
            raise KeyError(f"column {pt} or pseudonym not found")
    if column not in df.columns:
        raise KeyError(f"column {column} or pseudonym not found")

    grouped = df.groupby(on)
    drop : list = []
    clustered : list = []
    rows : list = []

    for _, group in grouped:
        cluster_entries = []
        for i, row in group.iterrows():
            cluster_entries.append(row[column])
        copy_row = copy.deepcopy(row)
        rows.append(copy_row)
        clustered.append(tuple(cluster_entries))

    df = pd.DataFrame(rows)
    index = df.columns.get_indexer_for([column])[0]
    df.drop(columns=[column], inplace=True)
    df.insert(int(index), column=column, value=clustered)
    return df

def compare(left: pd.DataFrame, right: pd.DataFrame, columns: str, on: str) -> dict:
    errors : dict = {"line":[], "column": [], "description": []}
    for pt in list(left[on]) + list(right[on]):
        matching_rows_left = left[left[on] == pt]
        matching_rows_right = right[right[on] == pt]
        for column in columns:
            for lc, rc in zip(matching_rows_left[column], matching_rows_right[column]):
                if lc != rc:
                    # filter out nan
                    if lc is np.nan and rc is np.nan:
                        continue
                    errors["line"].append(pt)
                    errors["column"].append(column)
                    errors["description"].append("{} ({}) != {} ({})".format(lc, type(lc), rc, type(rc)))
    return errors

def filter_df(df: pd.DataFrame, on: str, value, column: str, blank_defaults: bool) -> pd.DataFrame:
    if blank_defaults:
        filtered_df = df.loc[(df[on] == value) | (df[on].isnull())]
    else:
        filtered_df = df.loc[(df[on] == value)]
    return make_unique(filtered_df, column=column, prefer_column=on)

@click.group()
def gr1():
    pass

@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option("--on", type=str, multiple=True, required=True, help="Column to compare value")
@click.option("--column", type=str, required=True, help="Column to cluster into array")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command("cluster", help='''Cluster spreadsheet by column value''')
def cluster_command(spreadsheet, on, column, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    fname = os.path.split(os.path.splitext(spreadsheet)[0])[-1]
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    cluster(df, on, column).to_excel(f'{fname}_Clustered_On_{column}.xlsx', index=False)

@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option("--column", type=str, required=True, help="Clustered column")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command("uncluster", help='''Uncluster spreadsheet by column value''')
def uncluster_command(spreadsheet, column, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    fname = os.path.split(os.path.splitext(spreadsheet)[0])[-1]
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    uncluster(df, column).to_excel(f'{fname}_Unclustered_On_{column}.xlsx', index=False)

@click.option("-l", type=str, required=True, help="Left merge")
@click.option("-r", type=str, required=True, help="Right merge")
@click.option("--on", type=str, required=True, help="Column to merge on")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command(help='''Merge two spreadsheets on the given column''')
def merge(l, r, on, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    left = extract_columns_by_pseudonyms(read_file_to_df(l), pseudonyms)
    right = extract_columns_by_pseudonyms(read_file_to_df(r), pseudonyms)
    df = left.merge(right, on=on, how="left") # include all DNPs, unknown parts won't cause an error
    fname_l = os.path.split(os.path.splitext(l)[0])[-1]
    fname_r = os.path.split(os.path.splitext(r)[0])[-1]
    df.to_excel(f'{fname_l}_Merged{fname_r}_On_{on}.xlsx', index=False)

@click.option("-l", type=str, required=True, help="First spreadsheet")
@click.option("-r", type=str, required=True, help="Second spreadsheet")
@click.option("--on", type=str, default=None, help="Column to compare on")
@click.option("--columns", "-c", type=str, default=None, help="Columns to check, leave blank to check all with same name")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command("compare", help="Compares the given columns, passes if all given columns exist in both files and values are the same")
def compare_command(l, r, on, columns, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    left = extract_columns_by_pseudonyms(read_file_to_df(l), pseudonyms)
    right = extract_columns_by_pseudonyms(read_file_to_df(r), pseudonyms)
    if columns is None:
        columns = set(left.columns).intersection(set(right.columns))
    else:
        columns = [pt.strip() for pt in columns.split(",").strip(",")]

    errors = compare(left, right, columns, on)
    print("Comparing columns:", columns)
    for _, row in pd.DataFrame(errors).iterrows():
        print("[{}:{}] Comparison Failure: {}".format(row["column"], row["line"], row["description"]))

@click.option("--spreadsheet", "-s", type=str, required=True, help="Spreadsheet to filter from")
@click.option("--on", type=str, required=True, help="Column to compare on")
@click.option("--value", type=str, required=True, help="Value to select")
@click.option("--column", "-c", type=str, required=True, help="Column to use as primary value")
@click.option("--blank-defaults", is_flag=True, help="Include unmatched rows with no value in column")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command("filter", help="Compares the given columns, passes if all given columns exist in both files and values are the same")
def filter_command(spreadsheet, on, value, column, blank_defaults, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    filtered_df = filter_df(df, on, value, column, blank_defaults)
    fname = os.path.split(os.path.splitext(spreadsheet)[0])[-1]
    filtered_df.to_excel(f'{fname}_Filtered_On_{on}_{value}_by_{column}.xlsx', index=False)

@click.option("--fin", "-i", type=str, required=True, help="Input sreadsheet")
@click.option("--fout", "-o", type=str, help="Generatated spreadsheet")
@click.option("--format", type=click.Choice(supported_file_formats), help="Generatated spreadsheet format")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command("translate", help="Compares the given columns, passes if all given columns exist in both files and values are the same")
def translate_command(fin, fout, format, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)

    fname = os.path.split(os.path.splitext(fin)[0])[-1]
    if fout is None:
        if format is not None:
            fout = fname + format
        else:
            fout = fin

    df.to_excel(fout, index=False)

def main():
    gr1()

if __name__ == "__main__":
    main()
