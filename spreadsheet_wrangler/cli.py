"""Console script for spreadsheet_wrangler."""

import re
from pathlib import Path

import click
import pandas as pd

from . import cluster
from . import compare
from . import extract_columns_by_pseudonyms
from . import get_supported_file_formats
from . import read_file_to_df
from . import read_pseodonyms
from . import select_on_value
from . import uncluster
from . import write


@click.group()
def gr1():
    pass


@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option(
    "--on", type=str, multiple=True, required=True, help="Column to compare value"
)
@click.option("--column", type=str, required=True, help="Column to cluster into array")
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command("cluster", help="""Cluster spreadsheet by column value""")
def cluster_command(fout, spreadsheet, on, column, pseudonyms, format_):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    if fout is None:
        base = Path(spreadsheet).stem
        fname = f"{base}_Clustered_On_{column}.{format_}"
    else:
        fname = fout
    clustered_df = cluster(df, on, column)
    clustered_df[column] = [",".join(pt) for pt in clustered_df[column]]
    write(clustered_df, fname, index=False)


@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option("--column", type=str, required=True, help="Clustered column")
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command("uncluster", help="""Uncluster spreadsheet by column value""")
def uncluster_command(fout, spreadsheet, column, pseudonyms, format_):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    if fout is None:
        base = Path(spreadsheet).stem
        fname = f"{base}_Unclustered_On_{column}.{format_}"
    else:
        fname = fout
    write(uncluster(df, column), fname, index=False)


@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--left", "-l", type=str, required=True, help="Left merge")
@click.option("--right", "-r", type=str, required=True, help="Right merge")
@click.option("--on", type=str, required=True, help="Column to merge on")
@click.option(
    "--method",
    default="left",
    type=click.Choice(["left", "right", "outer", "inner"]),
    help="Column to merge on",
)
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command(help="""Merge two spreadsheets on the given column""")
def merge(fout, left, right, on, method, pseudonyms, format_):
    pseudonyms = read_pseodonyms(pseudonyms)
    left_df = extract_columns_by_pseudonyms(read_file_to_df(left), pseudonyms)
    right_df = extract_columns_by_pseudonyms(read_file_to_df(right), pseudonyms)
    df = left_df.merge(
        right_df, on=on, how=method
    )  # include all DNPs, unknown parts won't cause an error
    fname_l = Path(left).stem
    fname_r = Path(right).stem
    fname = f"{fname_l}_Merged{fname_r}_On_{on}.{format_}" if fout is None else fout
    write(df, fname, index=False)


@click.option("--left", "-l", type=str, required=True, help="First spreadsheet")
@click.option("--right", "-r", type=str, required=True, help="Second spreadsheet")
@click.option("--on", type=str, default=None, help="Column to compare on")
@click.option(
    "--columns",
    "-c",
    multiple=True,
    type=str,
    default=None,
    help="Columns to check, leave blank to check all with same name",
)
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@gr1.command(
    "compare",
    help="Compares the given columns, passes if all given columns exist in both files and values are the same",
)
def compare_command(left, right, on, columns, pseudonyms):
    pseudonyms = read_pseodonyms(pseudonyms)
    left_df = extract_columns_by_pseudonyms(read_file_to_df(left), pseudonyms)
    right_df = extract_columns_by_pseudonyms(read_file_to_df(right), pseudonyms)
    if columns is None:
        columns = set(left_df.columns).intersection(set(right_df.columns))

    errors = compare(left_df, right_df, columns, on)
    for _, _row in pd.DataFrame(errors).iterrows():
        pass


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--value", type=str, required=True, help="Value to select")
@click.option("--column", "-c", type=str, required=True, help="Column to filter on")
@click.option(
    "--blank-defaults",
    is_flag=True,
    help="Include unmatched rows with no value in column",
)
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command(
    "filter",
    help="Compares the given columns, selects the row if all given columns exist in both files and values are the same",
)
def filter_command(fin, fout, value, column, blank_defaults, pseudonyms, format_):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)
    filtered_df = select_on_value(
        df, column=column, value=value, blank_defaults=blank_defaults
    )
    if fout is None:
        base = Path(fin).stem
        fname = f"{base}_Filtered_On_{column}_by_{value}.{format_}"
    else:
        fname = fout
    write(filtered_df, fname, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command(
    "translate",
    help="Compares the given columns, passes if all given columns exist in both files and values are the same",
)
def translate_command(fin, fout, pseudonyms, format_):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)

    if fout is None:
        base = Path(fin).stem
        fout = f"{base}.{format_}" if format_ is not None else fin

    write(df, fout, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@gr1.command("to_md", help="Generate a markdown table")
def to_md(fin, fout):
    df = read_file_to_df(fin)
    with Path(fout).open("w") as f:
        f.write(df.to_markdown())


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option(
    "--column", "-c", type=str, required=True, help="Column to use as primary value"
)
@click.option("--delimiter", "-d", type=str, required=True, help="Current delimiter")
@click.option("--new-delimiter", "-n", type=str, required=True, help="New delimiter")
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command(
    "delimiter", help="Replace a delimiter with a new one, ex. replace tabs with commas"
)
def delimiter_command(fin, fout, column, delimiter, new_delimiter, pseudonyms, format_):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)
    col = []
    for _, line in df.iterrows():
        col.append(new_delimiter.join(line[column].split(delimiter)))
    df[column] = col

    if fout is None:
        base = Path(fin).stem
        fname = f"{base}_Delimiter_replaced_{column}.{format_}"
    else:
        fname = fout

    write(df, fname, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option(
    "--column", "-c", type=str, required=True, help="Column to use as primary value"
)
@click.option("--delimiter", "-d", type=str, required=True, help="Current delimiter")
@click.option(
    "--pseudonyms",
    "-p",
    type=str,
    default="",
    help="Alternative column names in json format",
)
@click.option(
    "--regex",
    "-r",
    type=str,
    default=r"(\(?\+?[0-9]*\)?)?[0-9_\- \(\)]",
    help="Regex expression to sort by, defaults to taking the first number",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command("sort-column", help="")
def sort_lines_in_column(fin, fout, column, delimiter, pseudonyms, regex, format_):
    """Break a line by delimiter, sort by number if found otherwise by string"""
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)
    col = []
    regex_compiled = re.compile(regex)
    for _, line in df.iterrows():
        entries = line[column].split(delimiter)
        try:
            try:
                sorted_list = sorted(
                    entries, key=lambda x: int(regex_compiled.search(x).group())
                )
            except ValueError:
                sorted_list = sorted(
                    entries, key=lambda x: regex_compiled.search(x).group()
                )
        except TypeError:
            sorted_list = sorted(entries)
        entry = delimiter.join(sorted_list)
        col.append(entry)
    df[column] = col
    if fout is None:
        base = Path(fin).stem
        fname = f"{base}_Sorted_{column}.{format_}"
    else:
        fname = fout
    write(df, fname, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--page", type=str, required=True, help="Page name to export")
@click.option(
    "--format",
    "format_",
    type=click.Choice(get_supported_file_formats()),
    default="xlsx",
    help="Generatated spreadsheet format",
)
@gr1.command("export-sheet", help="Export a single sheet of a spreadsheet")
def export_sheet(fin, fout, page, format_):
    """Export a single sheet of a spreadsheet"""
    path = Path(fin)
    base = path.stem

    fname = fout
    if fout is None:
        fname = f"{base}{page}.{format_}"

    if path.suffix.lower() == ".csv":  # no pages for a csv
        df = read_file_to_df(fin)
    else:
        df = pd.read_excel(fin, sheet_name=page)

    write(df, fname, index=False)


def main():
    gr1()


if __name__ == "__main__":
    main()
