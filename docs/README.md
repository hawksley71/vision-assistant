# Vision-Aware Smart Assistant

A smart assistant that uses computer vision to detect objects and answer questions about what it sees.

## Project Structure

```
vision-assistant/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── assistant.py   # Main detection and voice assistant
│   │   ├── voice_loop.py  # Voice interaction loop
│   │   ├── openai_assistant.py  # OpenAI integration
│   │   └── tts.py        # Text-to-speech utilities
│   ├── models/            # ML models
│   │   ├── yolov8_model.py
│   │   └── base_model.py
│   ├── utils/             # Utility functions
│   │   ├── audio.py      # Audio device handling
│   │   ├── combine_logs.py
│   │   └── openai_utils.py
│   └── config/            # Configuration
│       └── settings.py
├── data/                  # Data directory (not in git)
│   ├── raw/              # Raw detection logs
│   └── processed/        # Processed logs
├── docs/                  # Documentation
│   ├── reports/          # Project reports (not in git)
│   ├── data_setup.md     # Data setup guide
│   ├── windows_setup.md  # Windows setup guide
│   └── home_assistant_setup.md  # Home Assistant setup guide
├── models/               # Model weights (not in git)
│   └── yolov8/          # YOLOv8 weights
├── tools/               # Utility scripts
│   └── generate_fake_logs.py
└── requirements.txt     # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Webcam
- Microphone
- Docker (for Home Assistant)
- Internet connection (required for TTS and model downloads)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hawksley71/vision-assistant.git
   cd vision-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your data:
   - See [Data Setup Guide](docs/data_setup.md) for instructions
   - Generate test data using the provided scripts

5. Set up Home Assistant:
   - See [Home Assistant Setup Guide](docs/home_assistant_setup.md) for Docker installation and configuration
   - This is required for text-to-speech functionality
   - We recommend using Nabu Casa Cloud for reliable TTS and easy backup/restore

6. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Home Assistant token:
     ```
     HOME_ASSISTANT_TOKEN=your_token_here
     HOME_ASSISTANT_URL=http://localhost:8123
     HOME_ASSISTANT_TTS_SERVICE=tts.cloud_say  # If using Nabu Casa Cloud
     ```

## Usage

Run the assistant:
```bash
python -m src.core.voice_loop
```

## Features

- Real-time object detection using YOLOv8
- Voice interaction
- Natural language querying of detection history
- Cross-platform support (Linux, Windows)
- Home Assistant integration for TTS

## Documentation

- [Data Setup Guide](docs/data_setup.md) - How to set up and generate test data
- [Windows Setup Guide](docs/windows_setup.md) - Windows-specific setup instructions
- [Home Assistant Setup Guide](docs/home_assistant_setup.md) - Docker setup and configuration
- [Models Guide](models/README.md) - Information about model weights and setup
- [Test Questions](docs/assistant_test_questions.txt) - Sample questions for testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Overview
This project is a modular, voice-driven, vision-aware assistant that integrates live object detection (YOLOv8), historical log analysis, and natural language voice interaction. The system can answer both live and historical queries about detected objects, using the current camera feed and combined detection logs. It features robust intent detection, partial/fuzzy matching, and seamless integration with Home Assistant for text-to-speech (TTS) output.

## Technologies Used
- **Python 3.10**
- **YOLOv8** for object detection
- **OpenAI API** for code interpreter and pattern analysis
- **SpeechRecognition** and **gTTS** for voice input/output
- **OpenCV** for camera and image processing
- **Home Assistant** for TTS and smart home integration
- **Pandas, NumPy, scikit-learn** for data analysis
- **Requests, python-dotenv** for API and environment management

## Home Assistant Integration
Home Assistant is an open-source home automation platform. In this project, it is used to play TTS responses on a smart speaker. The assistant sends HTTP requests to Home Assistant's TTS service, which then vocalizes responses to the user.

## Setup Instructions

### Linux Setup
1. Install [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) or [Conda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment:
   ```
   mamba env create -f docs/environment.yml
   # or
   conda env create -f docs/environment.yml
   ```
3. Activate the environment:
   ```
   mamba activate vision-assistant
   # or
   conda activate vision-assistant
   ```
4. Set up your `.env` file with OpenAI and Home Assistant tokens.
5. Run the main assistant script as described below.

### Windows Setup
For Windows users, please refer to the detailed setup guide in `docs/windows_setup.md`. The guide covers:
- Prerequisites installation
- Project setup
- Audio configuration
- Home Assistant setup
- Common troubleshooting

## Codebase Structure and File Descriptions

### src/core/
- **assistant.py**: Main detection and voice assistant logic. Handles camera, detection buffer, query routing, and TTS output. Inputs: camera frames, voice input. Outputs: TTS responses, detection logs.
- **voice_loop.py**: Manages the main loop for voice and detection, including query classification and routing.
- **openai_assistant.py**: Handles OpenAI API integration for code interpreter and pattern analysis.
- **tts.py**: Utility for sending TTS messages to Home Assistant.

### src/utils/
- **audio.py**: Cross-platform microphone selection and audio device utilities. Supports both Linux and Windows systems.
- **openai_utils.py**: Helper functions for OpenAI API usage.
- **combine_logs.py**: Combines daily detection logs into a single CSV for historical analysis.

### src/models/
- **yolov8_model.py**: Wrapper for YOLOv8 object detection model.
- **yolov5_model.py**: Wrapper for YOLOv5 object detection model.
- **base_model.py**: Base class for detection models.

### src/config/
- **settings.py**: Centralized configuration for paths, camera, logging, audio, and Home Assistant.

### tools/
- **generate_fake_logs.py**: Generates synthetic detection logs for testing.
- **estimate_token_usage.py**: Estimates OpenAI API token usage for logs.

### Data Organization
- **data/raw/**: Original detection logs
- **data/processed/**: Combined and processed detection logs
- **data/logs/**: Application logs

### Other
- **docs/assistant_test_questions.txt**: List of test questions for all objects.
- **docs/environment.yml**: Environment specification for reproducibility.
- **docs/windows_setup.md**: Detailed Windows setup guide.

## What Has Been Accomplished
- Live object detection with YOLOv8 and robust detection buffer.
- Voice interaction with intent detection, partial/fuzzy matching, and pronoun resolution.
- Historical log analysis and pattern mining using OpenAI code interpreter.
- Seamless TTS output via Home Assistant.
- Modular, extensible codebase with clear separation of concerns.
- Comprehensive test question set for all objects.
- Cross-platform support (Linux and Windows).
- Standardized data organization structure.

(See `docs/reports/` for project plans, requirements, and summaries.)

## Possible Improvements and Future Directions
- Add a web dashboard for real-time and historical visualization.
- Improve pattern mining with more advanced ML/statistical methods.
- Support for multiple languages and voices.
- Integrate with more smart home devices (lights, sensors, etc.).
- Add user authentication and personalized responses.
- Optimize for edge devices (e.g., Raspberry Pi with Coral/Jetson).
- Expand dataset and detection classes for broader use cases.
- Add macOS support and documentation.

---

## How to Run
1. Start Home Assistant and ensure the TTS service is available.
2. Run the main assistant script:
   ```bash
   python -m src.core.voice_loop
   ```
3. Interact via microphone and listen for responses on your Home Assistant speaker.

---

## Contact
For questions or contributions, please see the project repository or contact the maintainer. 