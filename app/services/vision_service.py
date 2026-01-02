"""
Service module for visual analysis and captioning.

This module handles frame extraction from videos, calculates visual differences
between scenes to optimize processing, and interfaces with OpenAI's GPT-4o
model to generate descriptive captions for video frames.
"""

import base64
import os
import sys
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI

# Ensure project root is in sys.path for standalone execution
try:
    from app.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.config import Config

# Initialize OpenAI client using the centralized configuration
client = OpenAI(api_key=Config.OPENAI_API_KEY)


def extract_frame_at_time(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """
    Extracts a specific frame from the video at the given timestamp.

    Args:
        video_path (str): The absolute path to the video file.
        timestamp (float): The time in seconds to extract the frame from.

    Returns:
        Optional[np.ndarray]: The video frame as a NumPy array (OpenCV format),
                              or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)

    # Set position in milliseconds (OpenCV expects ms)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    success, frame = cap.read()
    cap.release()

    if not success:
        return None
    return frame


def get_frame_difference(frame1: Optional[np.ndarray], frame2: Optional[np.ndarray]) -> float:
    """
    Calculates the visual difference between two frames using Mean Squared Error.
    
    This is used to determine if a scene has changed significantly enough to
    warrant a new API call for captioning.

    Args:
        frame1 (Optional[np.ndarray]): The previous frame.
        frame2 (Optional[np.ndarray]): The current frame.

    Returns:
        float: A score representing the difference (0.0 is identical).
               Returns float('inf') if either frame is None.
    """
    if frame1 is None or frame2 is None:
        return float('inf')

    # Resize frames to a small resolution (64x64) for faster comparison
    target_size = (64, 64)
    f1_small = cv2.resize(frame1, target_size)
    f2_small = cv2.resize(frame2, target_size)

    # Convert to grayscale to focus on structural/luminance differences
    gray1 = cv2.cvtColor(f1_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2_small, cv2.COLOR_BGR2GRAY)

    # Calculate Mean Squared Error (MSE)
    err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
    err /= float(gray1.shape[0] * gray1.shape[1])
    
    return err


def encode_image_to_base64(frame: np.ndarray) -> str:
    """
    Encodes an OpenCV image frame to a Base64 string for API transmission.

    Args:
        frame (np.ndarray): The image frame to encode.

    Returns:
        str: The Base64 encoded string of the image.
    """
    # Encode frame to JPEG format
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Convert bytes to base64 string
    return base64.b64encode(buffer).decode('utf-8')


def generate_visual_caption(frame: np.ndarray) -> str:
    """
    Sends an image frame to the OpenAI API to generate a text description.

    Args:
        frame (np.ndarray): The image frame to describe.

    Returns:
        str: A generated caption describing the scene, entities, and actions.
    """
    base64_image = encode_image_to_base64(frame)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                "Analyze this news video frame for a search archive. "
                                "1. Identify famous public figures (politicians, athletes) by name. "
                                "2. Describe the setting and specific action (e.g., 'speech at UN', 'goal celebration'). "
                                "3. Transcribe visible context from banners or chyron if relevant. "
                                "Be concise and factual. Do not state 'I cannot identify anyone' or similar negatives if no public figures are found; simply describe the scene."
                            )
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                }
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error during visual captioning: {e}")
        return "Error analyzing image."


if __name__ == "__main__":
    # Test block to verify visual processing logic
    if os.path.exists(Config.VIDEO_DIR):
        test_videos = [
            os.path.join(Config.VIDEO_DIR, f) 
            for f in os.listdir(Config.VIDEO_DIR) 
            if f.endswith(".mp4")
        ]

        if test_videos:
            test_video_path = test_videos[0]
            print(f"Testing visual analysis on: {test_video_path}")
            
            # Extract a frame at the 10-second mark
            frame_10s = extract_frame_at_time(test_video_path, 10.0)
            
            if frame_10s is not None:
                print("Frame extracted successfully.")
                
                # Generate a caption for this frame
                print("Generating caption (this calls the OpenAI API)...")
                caption = generate_visual_caption(frame_10s)
                print(f"Generated Caption: {caption}")
            else:
                print("Failed to extract frame.")
        else:
            print(f"No videos found in {Config.VIDEO_DIR}")
    else:
        print(f"Video directory not found: {Config.VIDEO_DIR}")