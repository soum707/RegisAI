# ConversAI
This program records audio, transcribes it using whisper, enters the transcription as a prompt to chat gpt, and uses tts to speak it out loud. 

## Downloading Necessary Packages
```pip install openai```
```pip install whisper```
```pip install soundfile```
```pip install sounddevice```
```pip install numpy```
```pip install webrtcvad```
```pip install getts pygame```
```brew install ffmpeg```

## Usage
* Get your openai API key from "https://platform.openai.com/"
* Uncomment line 18 in main.py, and add replace your_api_key with the api key you got from openai

## Uses
* Say "Exit Conversation" to end the conversation
* Say "Type my prompt" to input your prompt using your keyboard

## Configerations 
* Changing ChatGPT model: Line 60, possible models ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview")
* Changing TTS model: line 71, possible models ("tts-1", "tts-1-hd")
* Changing TTS voice: line 72, possible voices ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
