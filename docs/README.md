# Vision-Aware Smart Assistant

## Project Overview
This project is a modular, voice-driven, vision-aware assistant that integrates live object detection (YOLOv8), historical log analysis, and natural language voice interaction. The system can answer both live and historical queries about detected objects, using the current camera feed and combined detection logs. It features robust intent detection, partial/fuzzy matching, and seamless integration with Home Assistant for text-to-speech (TTS) output.

## Technologies Used
- **Python 3.10**
- **YOLOv8/YOLOv5** for object detection
- **OpenAI API** for code interpreter and pattern analysis
- **SpeechRecognition** and **gTTS** for voice input/output
- **OpenCV** for camera and image processing
- **Home Assistant** for TTS and smart home integration
- **Pandas, NumPy, scikit-learn** for data analysis
- **Requests, python-dotenv** for API and environment management

## Home Assistant Integration
Home Assistant is an open-source home automation platform. In this project, it is used to play TTS responses on a smart speaker. The assistant sends HTTP requests to Home Assistant's TTS service, which then vocalizes responses to the user.

## Setup Instructions
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

## Codebase Structure and File Descriptions

### src/core/
- **assistant.py**: Main detection and voice assistant logic. Handles camera, detection buffer, query routing, and TTS output. Inputs: camera frames, voice input. Outputs: TTS responses, detection logs.
- **voice_loop.py**: Manages the main loop for voice and detection, including query classification and routing.
- **openai_assistant.py**: Handles OpenAI API integration for code interpreter and pattern analysis.
- **tts.py**: Utility for sending TTS messages to Home Assistant.
- **webhook.py**: (If used) Handles webhook integration for external triggers.

### src/utils/
- **audio.py**: Microphone selection and audio device utilities.
- **mic_test.py**: Script to test and debug microphone input.
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

### Other
- **docs/assistant_test_questions.txt**: List of test questions for all objects.
- **docs/environment.yml**: Environment specification for reproducibility.

## What Has Been Accomplished
- Live object detection with YOLOv8 and robust detection buffer.
- Voice interaction with intent detection, partial/fuzzy matching, and pronoun resolution.
- Historical log analysis and pattern mining using OpenAI code interpreter.
- Seamless TTS output via Home Assistant.
- Modular, extensible codebase with clear separation of concerns.
- Comprehensive test question set for all objects.

(See `docs/reports/` for project plans, requirements, and summaries.)

## Possible Improvements and Future Directions
- Add a web dashboard for real-time and historical visualization.
- Improve pattern mining with more advanced ML/statistical methods.
- Support for multiple languages and voices.
- Integrate with more smart home devices (lights, sensors, etc.).
- Add user authentication and personalized responses.
- Optimize for edge devices (e.g., Raspberry Pi with Coral/Jetson).
- Expand dataset and detection classes for broader use cases.

---

## How to Run
1. Start Home Assistant and ensure the TTS service is available.
2. Run the main assistant script (see `src/core/assistant.py` or `src/core/voice_loop.py`).
3. Interact via microphone and listen for responses on your Home Assistant speaker.

---

## Contact
For questions or contributions, please see the project repository or contact the maintainer. 