'''
spreadsheet_wrangler.py
Version 0.1.1
'''
import click
import pandas as pd  # type: ignore
import csv
import os
import numpy as np
import copy
import ast
import json

def read_pseodonyms(string):
    if len(string.strip()) == 0:
        return {}
    return json.loads(string)

'''Finds knowns pseudonyms for columns and includes names them correctly for passing as argument'''
def extract_columns_by_pseudonyms(df, column_names):
    included = []
    for name in df.columns:
        for column, names in column_names.items():
            if name.lower() in [pt.lower() for pt in names] or name.lower() == column.lower():
                df.rename(columns={name:column}, inplace=True)
    return df

def read_csv_to_df(fname: str) -> dict:
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

#sep=None
def read_file_to_df(fname: str) -> dict:
    name, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext == ".csv":
        df = read_csv_to_df(fname)

    elif ext in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods",".odt"]:
        df = pd.read_excel(fname, sheet_name=0, header=0, skiprows=0,
                comment="#", skip_blank_lines=True)
    return df

def uncluster(df: pd.DataFrame, grouped_column: str) -> pd.DataFrame:
    formated_df = df[df[grouped_column] != np.nan]
    expanded_rows = []
    for _, row in formated_df.iterrows():
        group = ast.literal_eval(row[grouped_column])
        for item in group:
            row[grouped_column] = item
            expanded_rows.append(copy.deepcopy(row))
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df

'''ref-des will not be a tuple of all the matching lines, the rest of the line is taken to be the first in the file and carried forward'''
def cluster(df: pd.DataFrame, on: str, column: str) -> pd.DataFrame:
    assert(on in df.columns)
    assert(column in df.columns)
    grouped = df.groupby(on)
    drop = []
    clustered = []
    rows = []
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


@click.group()
def gr1():
    pass

@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option("--on", type=str, required=True, help="Column to compare value")
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

def compare(left, right, columns, on):
    errors = {"line":[], "column": [], "description": []}
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


def main():
    gr1()

if __name__ == "__main__":
    main()
