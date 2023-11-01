"""Console script for spreadsheet_wrangler."""
import sys
import click
from . import *
from . import get_supported_file_formats

@click.group()
def gr1():
    pass


@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option("--on", type=str, multiple=True, required=True, help="Column to compare value")
@click.option("--column", type=str, required=True, help="Column to cluster into array")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("cluster", help='''Cluster spreadsheet by column value''')
def cluster_command(fout, spreadsheet, on, column, pseudonyms, format):
    pseudonyms=read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    if fout is None:
        base = os.path.split(os.path.splitext(spreadsheet)[0])[-1]
        fname = f'{base}_Clustered_On_{column}.{format}'
    else:
        fname = fout
    clustered_df = cluster(df, on, column)
    clustered_df[column] = [",".join(pt) for pt in clustered_df[column]]
    write(clustered_df, fname, index=False)


@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--spreadsheet", "-s", type=str, required=True, help="Main spreadsheet")
@click.option("--column", type=str, required=True, help="Clustered column")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("uncluster", help='''Uncluster spreadsheet by column value''')
def uncluster_command(fout, spreadsheet, column, pseudonyms, format):
    pseudonyms=read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(spreadsheet), pseudonyms)
    if fout is None:
        base = os.path.split(os.path.splitext(spreadsheet)[0])[-1]
        fname = f'{base}_Unclustered_On_{column}.{format}'
    else:
        fname = fout
    write(uncluster(df, column), fname, index=False)


@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("-l", type=str, required=True, help="Left merge")
@click.option("-r", type=str, required=True, help="Right merge")
@click.option("--on", type=str, required=True, help="Column to merge on")
@click.option("--method", default="left", type=click.Choice(["left", "right", "outer", "inner"]), help="Column to merge on")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command(help='''Merge two spreadsheets on the given column''')
def merge(fout, l, r, on, method, pseudonyms, format):
    pseudonyms=read_pseodonyms(pseudonyms)
    left = extract_columns_by_pseudonyms(read_file_to_df(l), pseudonyms)
    right = extract_columns_by_pseudonyms(read_file_to_df(r), pseudonyms)
    df = left.merge(right, on=on, how=method) # include all DNPs, unknown parts won't cause an error
    fname_l = os.path.split(os.path.splitext(l)[0])[-1]
    fname_r = os.path.split(os.path.splitext(r)[0])[-1]
    if fout is None:
        fname = f'{fname_l}_Merged{fname_r}_On_{on}.{format}'
    else:
        fname = fout
    write(df, fname, index=False)


@click.option("-l", type=str, required=True, help="First spreadsheet")
@click.option("-r", type=str, required=True, help="Second spreadsheet")
@click.option("--on", type=str, default=None, help="Column to compare on")
@click.option("--columns", "-c", multiple=True, type=str, default=None, help="Columns to check, leave blank to check all with same name")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@gr1.command("compare", help="Compares the given columns, passes if all given columns exist in both files and values are the same")
def compare_command(l, r, on, columns, pseudonyms):
    pseudonyms=read_pseodonyms(pseudonyms)
    left = extract_columns_by_pseudonyms(read_file_to_df(l), pseudonyms)
    right = extract_columns_by_pseudonyms(read_file_to_df(r), pseudonyms)
    if columns is None:
        columns = set(left.columns).intersection(set(right.columns))

    errors = compare(left, right, columns, on)
    print("Comparing columns:", columns)
    for _, row in pd.DataFrame(errors).iterrows():
        print("[{}:{}] Comparison Failure: {}".format(row["column"], row["line"], row["description"]))


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--value", type=str, required=True, help="Value to select")
@click.option("--column", "-c", type=str, required=True, help="Column to filter on")
@click.option("--blank-defaults", is_flag=True, help="Include unmatched rows with no value in column")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("filter", help="Compares the given columns, selects the row if all given columns exist in both files and values are the same")
def filter_command(fin, fout, value, column, blank_defaults, pseudonyms, format):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)
    filtered_df = select_on_value(df, column=column, value=value, blank_defaults=blank_defaults)
    if fout is None:
        base = os.path.split(os.path.splitext(fin)[0])[-1]
        fname = f'{base}_Filtered_On_{column}_by_{value}.{format}'
    else:
        fname = fout
    write(filtered_df, fname, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("translate", help="Compares the given columns, passes if all given columns exist in both files and values are the same")
def translate_command(fin, fout, pseudonyms, format):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)

    if fout is None:
        base = os.path.split(os.path.splitext(fin)[0])[-1]
        if format is not None:
            fout = f"{base}.{format}"
        else:
            fout = fin

    write(df, fout, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@gr1.command("to_md", help="Generate a markdown table")
def to_md(fin, fout):
    df = read_file_to_df(fin)
    with open(fout, "w") as f:
        f.write(df.to_markdown())

@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--column", "-c", type=str, required=True, help="Column to use as primary value")
@click.option("--delimiter", "-d", type=str, required=True, help="Current delimiter")
@click.option("--new-delimiter", "-n", type=str, required=True, help="New delimiter")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("delimiter", help="Replace a delimiter with a new one, ex. replace tabs with commas")
def delimiter_command(fin, fout, column, delimiter, new_delimiter, pseudonyms, format):
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)
    col = []
    for _, line in df.iterrows():
        col.append(new_delimiter.join(line[column].split(delimiter)))
    df[column] = col

    if fout is None:
        base = os.path.split(os.path.splitext(fin)[0])[-1]
        fname = f'{base}_Delimiter_replaced_{column}.{format}'
    else:
        fname = fout

    write(df, fout, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--column", "-c", type=str, required=True, help="Column to use as primary value")
@click.option("--delimiter", "-d", type=str, required=True, help="Current delimiter")
@click.option("--pseudonyms", "-p", type=str, default="", help="Alternative column names in json format")
@click.option("--regex", "-r", type=str, default="(\(?\+?[0-9]*\)?)?[0-9_\- \(\)]", help="Regex expression to sort by, defaults to taking the first number")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("sort-column", help="")
def sort_lines_in_column(fin, fout, column, delimiter, pseudonyms, regex, format):
    '''Break a line by delimiter, sort by number if found otherwise by string'''
    pseudonyms = read_pseodonyms(pseudonyms)
    df = extract_columns_by_pseudonyms(read_file_to_df(fin), pseudonyms)
    col = []
    regex_compiled = re.compile(regex)
    for _, line in df.iterrows():
        entries = line[column].split(delimiter)
        try:
            try:
                sorted_list = sorted(entries, key=lambda x: int(regex_compiled.search(x).group()))
            except ValueError:
                sorted_list = sorted(entries, key=lambda x: regex_compiled.search(x).group())
        except TypeError:
            sorted_list = sorted(entries)
        entry = delimiter.join(sorted_list)
        col.append(entry)
    df[column] = col
    if fout is None:
        base = os.path.split(os.path.splitext(fin)[0])[-1]
        fname = f'{base}_Sorted_{column}.{format}'
    else:
        fname = fout
    write(df, fout, index=False)


@click.option("--fin", "-i", type=str, required=True, help="Input spreadsheet")
@click.option("--fout", "-o", type=str, default=None, help="Generatated spreadsheet")
@click.option("--page", type=str, required=True, help="Page name to export")
@click.option("--format", type=click.Choice(get_supported_file_formats()), default="xlsx", help="Generatated spreadsheet format")
@gr1.command("export-sheet", help="Export a single sheet of a spreadsheet")
def export_sheet(fin, fout, page, format):
    '''Export a single sheet of a spreadsheet'''
    path, ext = os.path.splitext(fin)
    _, base = os.path.split(path)

    if fout is None:
        fname = f'{base}{page}.{format}'
    else:
        fname = fout

    if ext.lower() == "csv":  # no pages for a csv
        df = read_file_to_df(fin)
    else:
        df = pandas.read_excel(fin, sheet_name=page)

    write(df, fout, index=False)


def main():
    gr1()


if __name__ == "__main__":
    main()
