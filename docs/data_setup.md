# Data Setup Guide

This guide explains how to set up the data structure for the Vision-Aware Smart Assistant project.

## Data Directory Structure

The project expects the following data structure:

```
data/
├── raw/                    # Raw detection logs
│   ├── 2024-03-01.csv     # Daily detection logs
│   ├── 2024-03-02.csv
│   └── ...
├── processed/             # Processed and combined logs
│   └── combined_logs.csv  # Combined detection logs
└── audio/                 # Audio recordings
    ├── 2024-03-01/       # Daily audio files
    │   ├── 08:00:00.wav
    │   ├── 08:30:00.wav
    │   └── ...
    └── 2024-03-02/
        └── ...
```

## Setting Up Test Data

### 1. Create Required Directories

```bash
mkdir -p data/raw data/processed data/audio
```

### 2. Generate Test Detection Logs

The project includes a script to generate test detection logs. Run:

```bash
python scripts/generate_fake_logs.py
```

This will:
- Create sample detection logs in `data/raw/`
- Generate a combined log file in `data/processed/combined_logs.csv`
- Create realistic detection patterns for testing

### 3. Generate Test Audio Files (Optional)

If you want to test audio functionality:

```bash
python scripts/generate_fake_audio.py
```

This will create sample audio files in the `data/audio/` directory.

## Data Format

### Detection Logs
- Raw logs are stored as CSV files
- Each file contains detections for one day
- Format: `timestamp,object_class,confidence,location`

### Combined Logs
- Single CSV file with all detections
- Used for analysis and querying
- Automatically generated from raw logs

### Audio Files
- WAV format
- Organized by date and time
- Used for voice interaction testing

## Notes

- The `data/` directory is excluded from Git via `.gitignore`
- Each user needs to generate their own test data
- Real data can be used by placing files in the appropriate directories
- The project will automatically create the directory structure on first run

## Troubleshooting

If you encounter issues:

1. Check directory permissions
2. Ensure CSV files are properly formatted
3. Verify audio files are in WAV format
4. Check file naming conventions match the expected format

For more help, see the [main README](../README.md) or open an issue on GitHub. 