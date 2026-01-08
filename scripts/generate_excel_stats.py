from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
import io
import pandas as pd

def dataframe_to_formatted_excel(dfs: list, sheet_names:list, table_styles:list, output_filename: str = "formatted_data.xlsx", number_formats: dict = None):
    """
    Takes a pandas DataFrame and saves it to an Excel file as a clean,
    nicely formatted table object, with optional number formatting.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        output_filename (str): The name of the Excel file to save.
        number_formats (dict, optional): A dictionary where keys are column names
                                         and values are Excel number format strings
                                         (e.g., '0.00', '#,##0', '0.00E+00', '@' for text).
                                         Defaults to None.
    """
    try:
        # Create an Excel writer object using openpyxl engine
        writer = pd.ExcelWriter(output_filename, engine='openpyxl')
        for df, sheet_name, table_style in zip(dfs, sheet_names, table_styles):

            # Write the DataFrame to a specific sheet in the Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Access the workbook and the worksheet created by pandas
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Get column headers for mapping during cell iteration
            # The first row (index 1 in openpyxl) contains the headers
            column_headers = [cell.value for cell in worksheet[1]]

            # 1. Auto-adjust column widths
            for col_idx, col in enumerate(worksheet.columns):
                max_length = 0
                # get_column_letter is 1-indexed, so use col_idx + 1
                column_letter = get_column_letter(col_idx + 1)
                for cell in col:
                    try: # Handle case where cell.value is None or not a string
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2) * 1.2 # Add a little extra padding
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Define the range for the table
            # A1 is the top-left, and the bottom-right is determined by data dimensions
            max_row = len(df) + 1  # +1 for header row
            max_col = len(df.columns)
            table_ref = f"A1:{get_column_letter(max_col)}{max_row}"

            # Create a Table object
            # You can choose different table styles like 'TableStyleMedium9', 'TableStyleLight1' etc.
            # Check openpyxl documentation for available styles.
            table_name = f"{sheet_name.replace(' ', '')}Table"
            tab = Table(displayName=table_name, ref=table_ref)

            # Add a default style for the table
            style = TableStyleInfo(name=table_style, showFirstColumn=False,
                                showLastColumn=False, showRowStripes=True, showColumnStripes=False)
            tab.tableStyleInfo = style

            # Add the table to the worksheet
            worksheet.add_table(tab)

            # Apply custom number formats for data rows (excluding the header which is part of the table)
            # Iterating from min_row=2 to apply formats only to data cells
            for row_idx, row in enumerate(worksheet.iter_rows(min_row=2), start=2):
                for col_idx, cell in enumerate(row):
                    # Apply custom number format if specified for the column
                    if number_formats and column_headers and col_idx < len(column_headers):
                        column_name = column_headers[col_idx]
                        if column_name in number_formats:
                            cell.number_format = number_formats[column_name](cell.value)
                    # Align data left for all data cells, as table styling might center numbers by default
                    cell.alignment = Alignment(horizontal="left", vertical="top")

        # Save the Excel file
        writer.close()
        print(f"DataFrame successfully saved to '{output_filename}' as an Excel table with formatting.")

    except Exception as e:
        print(f"An error occurred: {e}")

format_dict = {
    'Feature': lambda x: '@',
    'PCC Rank': lambda x: '0',
    'PCC': lambda x: '0.00',
    'PCC pval': lambda x: '0.00E+00' if x < 0.01 else '0.00',
    'SRC Rank': lambda x: '0',
    'SRC': lambda x: '0.00',
    'SRC pval': lambda x: '0.00E+00' if x < 0.01 else '0.00',
    'MI Rank': lambda x: '0',
    'MI': lambda x: '0.00',
    'RFE Rank': lambda x: '0',
    'Avg Rank': lambda x: '0.00',
    'Type': lambda x: '@'
}

stats = pd.read_csv('data/processed/cof_feature_stats.csv', index_col=0)
stats_sample = pd.read_csv('data/processed/cof_brew_feature_stats.csv', index_col=0)
stats_judge = pd.read_csv('data/processed/cof_judge_feature_stats.csv', index_col=0)

out_stats = stats.reset_index().rename(columns={'index': 'Feature'}).sort_values(by='Avg Rank')
out_sample_stats = stats_sample.reset_index().rename(columns={'index': 'Feature'}).sort_values(by='Avg Rank')
out_judge_stats = stats_judge.reset_index().rename(columns={'index': 'Feature'}).sort_values(by='Avg Rank')
sheet_names = [
    'No aggregation',
    'Brew Score Aggregation',
    'Judge Score Aggregation',
]
table_styles = [
    'TableStyleMedium9',
    'TableStyleMedium10',
    'TableStyleMedium11',
]
dataframe_to_formatted_excel([out_stats, out_sample_stats, out_judge_stats], sheet_names, table_styles, 'stats.xlsx', format_dict)