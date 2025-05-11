import sys
import os
import pandas as pd

def load_all_logs(log_dir="data/raw"):
    """
    Combine all CSV log files in the specified directory into a single DataFrame.
    Args:
        log_dir (str): Directory containing CSV log files.
    Returns:
        pd.DataFrame: Combined DataFrame of all logs.
    """
    all_dfs = []
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
    for f in log_files:
        try:
            df = pd.read_csv(os.path.join(log_dir, f))
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Skipped {f} due to read error: {e}")
    if not all_dfs:
        print(f"No CSV log files found in {log_dir}.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")

if __name__ == "__main__":
    # Usage: python combine_logs_to_csv.py [log_dir] [output_csv]
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "outputs/combined_logs.csv"
    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        sys.exit(1)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = load_all_logs(log_dir)
    if df.empty:
        print("No data to save.")
        sys.exit(0)
    df.to_csv(output_csv, index=False)
    print(f"Combined DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Saved combined logs to: {output_csv}") 