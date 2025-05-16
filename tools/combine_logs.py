import os
import pandas as pd
from src.core.assistant import DetectionAssistant

def combine_logs():
    assistant = DetectionAssistant(None)
    df = assistant.load_all_logs()
    if df.empty:
        print("No logs found to combine.")
        return
    output_path = os.path.join('data', 'combined_logs.csv')
    df.to_csv(output_path, index=False)
    print(f"Combined logs written to {output_path}")

if __name__ == "__main__":
    combine_logs() 