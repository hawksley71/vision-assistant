from setuptools import setup, find_packages

setup(
    name="vision-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "torch",
        "ultralytics",
        "gtts",
        "SpeechRecognition",
        "pandas",
        "python-dotenv",
        "requests",
    ],
    python_requires=">=3.8",
) 