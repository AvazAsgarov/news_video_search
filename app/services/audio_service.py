"""
Service module for audio extraction and transcription.

This module handles the extraction of audio tracks from video files and
interfaces with the OpenAI Whisper API to generate time-stamped transcripts.
"""

import os
import sys
from typing import List, Optional, Any

from moviepy import VideoFileClip
from openai import OpenAI

# Ensure project root is in sys.path for standalone execution
try:
    from app.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.config import Config

# Initialize OpenAI client using the centralized configuration
client = OpenAI(api_key=Config.OPENAI_API_KEY)


def extract_audio(video_path: str) -> Optional[str]:
    """
    Extracts the audio track from a video file and saves it as an MP3.

    Args:
        video_path (str): The absolute path to the source video file.

    Returns:
        Optional[str]: The path to the generated MP3 file, or None if the
                       extraction fails.
    """
    try:
        # Derive filenames
        video_filename = os.path.basename(video_path)
        base_name = os.path.splitext(video_filename)[0]
        audio_filename = f"{base_name}.mp3"
        
        # Construct full path using the configured temp directory
        audio_path = os.path.join(Config.TEMP_AUDIO_DIR, audio_filename)

        # Optimization: Check if audio already exists to skip redundant processing
        if os.path.exists(audio_path):
            return audio_path

        print(f"Extracting audio from: {video_filename}")
        
        # Load video and write audio file
        # logger=None suppresses the default progress bar to keep logs clean
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3', logger=None)
        
        # Explicitly close the file handle to prevent file locking issues on Windows
        video.close()

        return audio_path

    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def transcribe_audio(audio_path: str) -> Optional[List[Any]]:
    """
    Transcribes an audio file using OpenAI's Whisper model.

    Args:
        audio_path (str): The path to the MP3 file to transcribe.

    Returns:
        Optional[List[Any]]: A list of transcript segments containing text
                             and start/end timestamps. Returns None on failure.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None

    print(f"Transcribing audio file: {os.path.basename(audio_path)}")

    try:
        # Open file in binary read mode
        # Using 'with' ensures the file is closed immediately after the API call
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        print("Transcription completed successfully.")
        return transcript.segments

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


if __name__ == "__main__":
    # Test block to verify service functionality
    # Looks for any MP4 file in the configured video directory
    if os.path.exists(Config.VIDEO_DIR):
        test_videos = [
            os.path.join(Config.VIDEO_DIR, f) 
            for f in os.listdir(Config.VIDEO_DIR) 
            if f.endswith(".mp4")
        ]

        if test_videos:
            test_video_path = test_videos[0]
            print(f"Testing with video: {test_video_path}")

            # 1. Test Extraction
            extracted_audio = extract_audio(test_video_path)
            
            if extracted_audio:
                print(f"Audio extracted to: {extracted_audio}")
                
                # 2. Test Transcription
                segments = transcribe_audio(extracted_audio)
                if segments:
                    print(f"Received {len(segments)} segments.")
                    print(f"First segment text: {segments[0].text}")
        else:
            print(f"No videos found in {Config.VIDEO_DIR} for testing.")
    else:
        print(f"Video directory does not exist: {Config.VIDEO_DIR}")