import sys
import os
import pandas as pd

try:
    import tiktoken
except ImportError:
    print("tiktoken is not installed. Please install it with 'pip install tiktoken'.")
    sys.exit(1)

def estimate_tokens(text, model="gpt-3.5-turbo"):
    """
    Estimate the number of tokens in a string for a given OpenAI model.
    Args:
        text (str): The text to estimate tokens for.
        model (str): The OpenAI model name (default: 'gpt-3.5-turbo').
    Returns:
        int: The estimated number of tokens.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

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
    # Usage: python estimate_token_usage.py [log_dir] [model]
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-3.5-turbo"
    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        sys.exit(1)
    df = load_all_logs(log_dir)
    if df.empty:
        print("No data to estimate.")
        sys.exit(0)
    csv_string = df.to_csv(index=False)
    num_tokens = estimate_tokens(csv_string, model)
    print(f"Combined DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Estimated tokens for all logs in '{log_dir}' (model: {model}): {num_tokens}") 