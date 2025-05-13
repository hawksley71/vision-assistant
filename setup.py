from setuptools import setup, find_packages

setup(
    name="vision-assistant",
    version="0.1.0",
    description="A vision-aware smart assistant with object detection and voice interaction",
    author="Hawksley",
    author_email="hawksley71@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "scikit-learn",
        "requests",
        "python-dotenv",
        "gtts",
        "SpeechRecognition",
        "openai",
        "PyAudio",
        "ultralytics",  # for YOLOv8
        "python-dateutil",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 