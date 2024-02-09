import sounddevice as sd
import numpy as np
import soundfile as sf
import whisper
from openai import OpenAI
import openai
import webrtcvad
import collections
from gtts import gTTS
import pygame
import io
import tempfile
import os
import interpreter

client = OpenAI()
# api_key = 'your_api_key'

def record_audio(vad, fs=16000, frame_duration=30, padding_duration=300):
    """ Record audio until speech stops based on VAD. """
    num_padding_frames = padding_duration // frame_duration
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    vad_frames = []
    print("Say your command (auto-detects end of speech):")

    with sd.RawInputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        while True:
            frame, _ = stream.read(int(fs * frame_duration / 1000))
            is_speech = vad.is_speech(frame, fs)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    vad_frames.extend(frame for frame, _ in ring_buffer)
                    ring_buffer.clear()
            else:
                vad_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    break

    return b''.join(vad_frames)

def transcribe_audio(filename, task, language=None):
    model = whisper.load_model("base")  
    print("Transcribing...")
    result = model.transcribe(filename, task=task, language=language)
    return result


def chat_with_gpt(messages):
    # openai.api_key = api_key  # Replace with your OpenAI API key
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def speak_text(text):
    """ Convert text to speech and play it using OpenAI. """
    # Set your OpenAI API key
    # openai.api_key = api_key  # Replace with your OpenAI API key

    # Generate spoken audio from input text
    response = openai.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    # Save the audio file to a temporary file
    fp = tempfile.NamedTemporaryFile(delete=False)
    response.stream_to_file(fp.name)
    fp.close()

    # Initialize and play audio
    pygame.mixer.init()
    pygame.mixer.music.load(fp.name)
    pygame.mixer.music.play()

    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Clean up temporary file
    os.remove(fp.name)


# Initialize VAD
vad = webrtcvad.Vad(1)  # 1: Moderate filtering

# Initialize conversation context
conversation = []
use_text_input = False

while True:
    if use_text_input:
        # Get user input via text
        user_message = input("Type your message: ")
        use_text_input = False  # Reset the flag
    else:
        # Record and process user message via voice
        audio_data = record_audio(vad)
        filename = "recording.wav"
        sf.write(filename, np.frombuffer(audio_data, dtype=np.int16), 16000)
        task = "speech-recognition"
        language = "en"
        transcription_result = transcribe_audio(filename, task, language)
        user_message = transcription_result["text"]

        # Check if the user said "type my prompt"
        if "type my prompt" in user_message.lower():
            print("Switching to text input for the next command.")
            use_text_input = True
            continue  # Skip the rest of the loop and prompt for text input

    print("You:", user_message)

    # Check if the user wants to exit the conversation
    if "exit conversation" in user_message.lower():
        print("Exiting conversation.")
        break

    # Continue with processing the message
    conversation.append({"role": "user", "content": user_message})
    gpt_response = chat_with_gpt(conversation)
    print("GPT:", gpt_response)
    speak_text(gpt_response)  # Speak out the GPT response
    conversation.append({"role": "assistant", "content": gpt_response})