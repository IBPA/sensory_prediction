import pandas as pd
from glob import glob
import os

# Quick script to merge all grid search result CSVs into a single Excel file
excel_path = 'scripts/collated_grid_search_results.xlsx'
csv_files = glob('./models/**/**/grid_search_results_*.csv', recursive=True)

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        df = pd.read_csv(csv_file)
        parts = csv_file.split(os.path.sep)
        
        # Extract the model type and data aggregation level from the path,
        # must be >31 chars
        model_type = parts[-2]
        if model_type == 'classifiers': model_type = 'cls'
        if model_type == 'regressors': model_type = 'reg'
        data_agg = parts[-4].replace('aggregation', 'agg')
        sheet_name = f"{model_type}_{data_agg}"
        
        df.to_excel(writer, sheet_name=sheet_name, index=False)
