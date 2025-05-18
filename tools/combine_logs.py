import os
import pandas as pd
from src.config.settings import PATHS

def combine_logs():
    # Get the raw data directory
    raw_data_dir = PATHS['data']['raw']
    
    # Get all CSV files in the raw data directory
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No log files found in", raw_data_dir)
        return
    
    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        file_path = os.path.join(raw_data_dir, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Read {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid log files found to combine.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.sort_values('timestamp')
    
    # Save to combined logs file
    output_path = PATHS['data']['combined_logs']
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(combined_df)} rows from {len(dfs)} files")
    print(f"Combined logs written to {output_path}")

if __name__ == "__main__":
    combine_logs() 