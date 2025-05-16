from setuptools import setup, find_packages

# Read README.md safely
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except Exception as e:
    print(f"Warning: Could not read README.md: {str(e)}")
    long_description = "A vision-aware smart assistant using YOLOv8 and OpenAI"

setup(
    name="vision-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "scikit-learn",
        "requests",
        "python-dotenv",
        "gtts",
        "speechrecognition",
        "openai",
        "pyaudio",
        "ultralytics",
        "python-dateutil",
        "homeassistant-api",
        "python-homeassistant-api",
        "websockets",
        "aiohttp",
        "asyncio",
    ],
    author="Hawksley",
    author_email="",  # Removed for privacy
    description="A vision-aware smart assistant using YOLOv8 and OpenAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vision-assistant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
) 