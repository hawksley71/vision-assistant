# Windows Setup Guide

This guide will help you set up the Vision-Aware Smart Assistant on a Windows system.

## Prerequisites

1. **Python Installation**:
   - Download and install Python 3.10 or higher from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Git Installation**:
   - Download and install Git from [git-scm.com](https://git-scm.com/download/win)
   - Verify installation: `git --version`

3. **Visual Studio Build Tools**:
   - Download and install from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - During installation, select "Desktop development with C++"
   - This is required for building some Python packages

4. **CUDA (Optional, for GPU support)**:
   - Download and install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
   - Verify installation: `nvidia-smi`

## Project Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hawksley71/vision-assistant.git
   cd vision-assistant
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyAudio**:
   - Download the appropriate PyAudio wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
   - Install using pip:
     ```bash
     pip install [downloaded-wheel-file].whl
     ```

5. **Install OpenCV**:
   ```bash
   pip install opencv-python
   ```

## Audio Setup

Windows doesn't use ALSA, so you'll need to modify the audio settings:

1. **Option 1: Use PyAudio's Windows Backend**:
   - Modify `src/utils/audio.py` to use the Windows backend
   - No ALSA installation needed

2. **Option 2: Install ALSA for Windows**:
   - Download and install ALSA from [here](https://www.alsa-project.org/main/index.php/Download)
   - This is more complex and not recommended

## Home Assistant Setup

1. **Option 1: Install Home Assistant**:
   - Follow the [Home Assistant Windows installation guide](https://www.home-assistant.io/installation/windows)
   - Default port is 8123

2. **Option 2: Modify Code for Different TTS**:
   - Edit `src/core/assistant.py` and `src/core/voice_loop.py`
   - Replace Home Assistant TTS with a different service

## Running the Assistant

1. **Set Environment Variables**:
   - Create a `.env` file in the project root
   - Add your Home Assistant token:
     ```
     HOME_ASSISTANT_TOKEN=your_token_here
     ```

2. **Run the Assistant**:
   ```bash
   python -m src.core.voice_loop
   ```

## Troubleshooting

1. **PyAudio Installation Issues**:
   - Make sure Visual Studio Build Tools are installed
   - Try installing the pre-built wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

2. **Camera Access Issues**:
   - Make sure your webcam is properly connected
   - Try running as administrator
   - Check Windows privacy settings for camera access

3. **Audio Issues**:
   - Check Windows sound settings
   - Make sure the correct microphone is selected
   - Try running as administrator

4. **CUDA Issues**:
   - Make sure NVIDIA drivers are up to date
   - Verify CUDA installation with `nvidia-smi`
   - Check PyTorch CUDA installation with:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```

## Additional Notes

- The code uses relative paths, so it should work on any drive
- File permissions are handled differently on Windows, but shouldn't cause issues
- If you encounter any path-related issues, use `os.path.normpath()` to normalize paths
- For development, consider using Visual Studio Code with Python extensions 