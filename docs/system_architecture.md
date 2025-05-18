```mermaid
graph TB
    %% Input Devices
    Camera[Camera Input]
    Microphone[Microphone Input]
    
    %% Core Components
    CV[Computer Vision Model]
    Assistant[Detection Assistant]
    PatternAnalyzer[Pattern Analyzer]
    OpenAI[OpenAI Client]
    
    %% Data Storage
    DetectionLogs[(Detection Logs)]
    PatternSummary[(Pattern Summary)]
    
    %% Output Devices
    Display[Display Output]
    HA[Home Assistant]
    Speaker[Smart Speaker]
    
    %% Input Device Connections
    Camera -->|Video Stream| CV
    Microphone -->|Audio Input| Assistant
    
    %% Core Component Connections
    CV -->|Detections| Assistant
    Assistant -->|Pattern Analysis| PatternAnalyzer
    Assistant -->|Query Processing| OpenAI
    
    %% Data Storage Connections
    Assistant -->|Log Detections| DetectionLogs
    PatternAnalyzer -->|Update Patterns| PatternSummary
    DetectionLogs -->|Historical Data| PatternAnalyzer
    
    %% Output Device Connections
    Assistant -->|Visual Feed| Display
    Assistant -->|Text Response| HA
    HA -->|TTS| Speaker
    
    %% Styling
    classDef input fill:#d4f1f9,stroke:#333,stroke-width:2px
    classDef core fill:#d5e8d4,stroke:#333,stroke-width:2px
    classDef storage fill:#ffe6cc,stroke:#333,stroke-width:2px
    classDef output fill:#fff2cc,stroke:#333,stroke-width:2px
    
    class Camera,Microphone input
    class CV,Assistant,PatternAnalyzer,OpenAI core
    class DetectionLogs,PatternSummary storage
    class Display,HA,Speaker output
```

# System Architecture

This diagram shows the high-level architecture of the vision-aware assistant system. Here's a brief explanation of each component:

## Input Devices
- **Camera**: Captures video stream for object detection
- **Microphone**: Captures voice commands and queries

## Core Components
- **Computer Vision Model**: Performs real-time object detection on video stream
- **Detection Assistant**: Main component that coordinates all other parts
- **Pattern Analyzer**: Analyzes detection patterns over time
- **OpenAI Client**: Processes natural language queries

## Data Storage
- **Detection Logs**: Stores historical detection data
- **Pattern Summary**: Stores analyzed patterns and trends

## Output Devices
- **Display**: Shows video feed with detection overlays
- **Text-to-Speech**: Converts responses to speech
- **Home Assistant**: Handles audio output and integration
- **Smart Speaker**: Plays the assistant's spoken responses

## Key Interactions
1. Camera feeds video to Computer Vision for detection
2. Detections are logged and analyzed for patterns
3. Voice commands are processed by the Assistant
4. Responses are converted to speech via Home Assistant and played on the Smart Speaker
5. Historical data is used to answer pattern-related queries 