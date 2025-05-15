# Home Assistant Setup Guide

This guide outlines the process for setting up Home Assistant with Docker for use with the Vision-Aware Smart Assistant.

## Overview

The Vision-Aware Smart Assistant uses Home Assistant for text-to-speech (TTS) output. This guide will help you:
1. Set up Home Assistant using Docker
2. Configure TTS services
3. Set up a media player for audio output
4. Generate and configure the required API token

## Prerequisites

- Docker installed on your system
- A media player (smart speaker, computer speakers, etc.)
- Basic understanding of Docker and networking
- Internet connection (required for TTS and model downloads)

## Setup Steps

### 1. Install Home Assistant with Docker

1. Create a directory for Home Assistant:
   ```bash
   mkdir -p ~/homeassistant
   ```

2. Create a Docker Compose file:
   ```yaml
   # ~/homeassistant/docker-compose.yml
   version: '3'
   services:
     homeassistant:
       container_name: homeassistant
       image: ghcr.io/home-assistant/home-assistant:stable
       volumes:
         - ./config:/config
         - /etc/localtime:/etc/localtime:ro
       restart: unless-stopped
       privileged: true
       network_mode: host
   ```

3. Start Home Assistant:
   ```bash
   cd ~/homeassistant
   docker-compose up -d
   ```

### 2. Initial Configuration

1. Access Home Assistant:
   - Open `http://localhost:8123` in your browser
   - Create your admin account
   - Complete the initial setup

2. Install required add-ons:
   - Go to Settings > Add-ons
   - Install "Google Translate TTS" or your preferred TTS service

### 3. Configure TTS and Media Player

1. Set up a media player:
   - Add your smart speaker or audio device
   - Note the entity ID (e.g., `media_player.living_room_speaker`)

2. Configure TTS service:
   - Go to Settings > Voice assistants
   - Enable and configure your chosen TTS service
   - Test the TTS service with your media player

### 4. Set up Nabu Casa Cloud (Recommended)

Nabu Casa Cloud provides several benefits:
- Reliable cloud TTS service
- Easy backup and restore between different machines
- Remote access to your Home Assistant instance
- Better TTS quality and reliability

1. Sign up for Nabu Casa Cloud:
   - Go to [Nabu Casa](https://www.nabucasa.com/)
   - Create an account
   - Subscribe to Home Assistant Cloud (free trial available)

2. Connect Home Assistant to Nabu Casa:
   - In Home Assistant, go to Settings > Home Assistant Cloud
   - Click "Connect" and follow the setup process
   - This will enable cloud TTS and remote access

3. Update your `.env` file to use cloud TTS:
   ```
   HOME_ASSISTANT_TOKEN=your_token_here
   HOME_ASSISTANT_URL=http://localhost:8123
   HOME_ASSISTANT_TTS_SERVICE=tts.cloud_say  # Use cloud TTS service
   ```

### 5. Generate API Token

1. Create a Long-Lived Access Token:
   - Go to your profile (click your username)
   - Scroll to the bottom
   - Create a token
   - Copy the token for use in the Vision-Aware Smart Assistant

2. Add the token to your `.env` file:
   ```
   HOME_ASSISTANT_TOKEN=your_token_here
   HOME_ASSISTANT_URL=http://localhost:8123
   ```

## Troubleshooting

### Common Issues

1. **Can't access Home Assistant**:
   - Check if Docker container is running: `docker ps`
   - Verify port 8123 is not in use
   - Check Docker logs: `docker logs homeassistant`

2. **TTS not working**:
   - Verify media player is properly configured
   - Check TTS service is installed and enabled
   - Test TTS directly in Home Assistant
   - If using cloud TTS, verify Nabu Casa connection

3. **API token issues**:
   - Ensure token is properly copied
   - Check token hasn't expired
   - Verify URL is correct in `.env`

4. **Cloud TTS issues**:
   - Check internet connection
   - Verify Nabu Casa subscription is active
   - Check Home Assistant Cloud connection status
   - Try restarting Home Assistant

## Additional Resources

- [Home Assistant Docker Installation](https://www.home-assistant.io/installation/linux#docker-compose)
- [Home Assistant TTS Documentation](https://www.home-assistant.io/integrations/tts/)
- [Home Assistant API Documentation](https://developers.home-assistant.io/docs/api/rest/)
- [Docker Documentation](https://docs.docker.com/)
- [Nabu Casa Documentation](https://www.nabucasa.com/config_entries/)
- [Home Assistant Cloud Documentation](https://www.home-assistant.io/integrations/cloud/)

## Next Steps

After setting up Home Assistant:
1. Update your `.env` file with the correct token and URL
2. Test the connection using the Vision-Aware Smart Assistant
3. Configure your preferred TTS voice and media player settings
4. If using Nabu Casa, test cloud TTS functionality 