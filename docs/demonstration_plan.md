# Demonstration Plan: Vision-Aware Smart Assistant

## 1. Introduction (2 min)
- Briefly introduce yourself and the project.
- State the goals: a modular, voice-driven, vision-aware assistant that can answer live and historical queries about detected objects, with Home Assistant integration.

## 2. Environment Setup (2 min)
- Show the environment.yml and how to create/activate the environment.
- Briefly show the .env file (with secrets hidden) and explain the need for OpenAI and Home Assistant tokens.
- Start Home Assistant and confirm TTS service is available.

## 3. Live Object Detection Demo (3 min)
- Start the assistant (run the main script).
- Show the live camera feed and detection buffer warming up.
- Point objects at the camera and show real-time detection in the terminal and on the feed.

## 4. Live Queries (3 min)
- Ask: "What do you see?" and demonstrate the assistant's response.
- Ask about confidence: "How sure are you?"
- Show the assistant's ability to answer about the current scene.

## 5. Historical Queries (3 min)
- Ask: "Have you seen a truck before?"
- Ask: "When did you last see a dog?"
- Show partial/fuzzy matching: "Have you seen mail?" (should suggest "mail truck").
- Demonstrate pronoun resolution: "Have you seen one before?" after identifying an object.

## 6. Pattern/Frequency Queries (3 min)
- Ask: "Is there a pattern for the school bus?"
- Ask: "When does the mail truck come?"
- Ask: "Does the garbage truck come on Tuesdays?"
- Show the assistant's ability to mine and report patterns from logs.

## 7. Home Assistant TTS Integration (2 min)
- Show the Home Assistant dashboard.
- Demonstrate TTS responses being played on the smart speaker.
- Optionally, show the TTS service call in Home Assistant logs.

## 8. Edge Cases and Robustness (1 min)
- Ask about objects not in the logs: "Have you seen a unicorn?"
- Ask ambiguous queries: "Have you seen it?"
- Show how the assistant handles and prompts for clarification.

## 9. Summary and Future Directions (1 min)
- Recap what was demonstrated.
- Briefly discuss possible improvements and future work (see README).
- Thank the viewer and provide contact info or repo link.

---

**Tips:**
- Use the test questions in `docs/assistant_test_questions.txt` for variety.
- Keep each section focused and concise.
- Show both successes and how the system handles errors or unknowns. 