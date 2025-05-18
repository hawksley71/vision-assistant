# Tools

This directory contains utility scripts for the Vision-Aware Smart Assistant project.

## Data Generation

The `generate_fake_logs.py` script creates realistic detection logs for testing the assistant. These logs simulate object detections with timestamps, counts, and confidence scores.

### Usage

```bash
# Generate 7 days of test data with 100 detections per day (default)
./generate_fake_logs.py

# Generate custom number of days and detections
python generate_fake_logs.py --days 14 --detections 200
```

### Output

The script generates CSV files in the `data/raw` directory with the following format:
- One file per day (e.g., `detections_2024_03_20.csv`)
- Each file contains:
  - Timestamp
  - Up to 3 detected objects per timestamp
  - Count and confidence score for each detection

### Example Output

```csv
timestamp,label_1,count_1,avg_conf_1,label_2,count_2,avg_conf_2,label_3,count_3,avg_conf_3
2024-03-20 08:15:23,person,2,0.923,car,1,0.856,,
2024-03-20 08:16:45,dog,1,0.912,,
2024-03-20 08:17:12,school bus,1,0.945,person,3,0.891,
```

### Notes

- The generated data includes a variety of common objects and their variations
- Timestamps are distributed throughout each day
- Confidence scores range from 0.5 to 0.95
- Object counts range from 1 to 5
- The data is suitable for testing the assistant's pattern recognition and query capabilities 