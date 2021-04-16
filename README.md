# spreadsheet-wrangler
Command line tool for interacting with spreadsheet data

## Functions
- merge: Left merge two spreadsheets and save as xlsx
- compare: Compare two spreadsheets on a column name, prints out the discrepencies
- cluster: Combine the same values in a specified column as an array with the same name as the clustered column. The remainder of the first rows data is kept.
- uncluster: Unpack clustered columns into one entry for each. The row is duplicated for each entry.
