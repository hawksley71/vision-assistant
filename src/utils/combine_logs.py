import os
import sys
import pandas as pd

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def combine_and_clean_logs(log_dir=None, output_file=None):
    """
    Combine all CSV detection logs in log_dir into a single cleaned CSV file.
    Args:
        log_dir (str): Directory containing raw log CSVs. If None, uses default data/raw in project root.
        output_file (str): Output CSV file path. If None, uses default outputs/combined_logs.txt in project root.
    Returns:
        pd.DataFrame: The combined and cleaned DataFrame.
    """
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "data/raw")
    print(f"Looking for logs in: {log_dir}")
    if output_file is None:
        output_file = os.path.join(PROJECT_ROOT, "outputs/combined_logs.csv")
    
    all_logs = []
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
    for fname in log_files:
        try:
            df = pd.read_csv(os.path.join(log_dir, fname))
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            all_logs.append(df)
        except Exception as e:
            print(f"Warning: Skipped {fname} due to read error: {e}")
    if not all_logs:
        print(f"No CSV files found in {log_dir}.")
        return pd.DataFrame()
    combined_df = pd.concat(all_logs, ignore_index=True)
    # Clean up: remove rows missing timestamp or label_1, fill NAs
    combined_df = combined_df.dropna(subset=['timestamp', 'label_1'])
    combined_df = combined_df.fillna('')
    combined_df = combined_df.sort_values("timestamp")
    # Write as standard CSV with comma separator
    combined_df.to_csv(output_file, index=False)
    print(f"Combined logs saved to {output_file} (shape: {combined_df.shape})")
    return combined_df

if __name__ == "__main__":
    # Usage: python combine_logs.py [log_dir] [output_csv]
    log_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    if log_dir and not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        sys.exit(1)
    os.makedirs(os.path.dirname(output_csv) or "outputs", exist_ok=True)
    df = combine_and_clean_logs(log_dir, output_csv)
    if df.empty:
        print("No data to save.")
        sys.exit(0)
    print(f"Combined DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Saved combined logs to: {output_csv}") 