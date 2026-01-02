"""
Core module for Optical Character Recognition (OCR).

This module uses EasyOCR to extract text from video frames, enabling the system
to index on-screen text such as news tickers, headlines, and banners.
"""

import os
import sys
from typing import Optional

import cv2
import easyocr
import numpy as np


# Initialize the EasyOCR Reader at the module level.
# This ensures the heavy models are loaded only once when the application starts,
# rather than reloading them for every single video frame.
# gpu=True attempts to use NVIDIA CUDA acceleration; it automatically falls back
# to CPU if no GPU is detected.
try:
    reader = easyocr.Reader(['en'], gpu=True)
except Exception as e:
    print(f"Warning: Failed to initialize EasyOCR. Error: {e}")
    # In a production environment, you might want to raise this error to stop execution,
    # but for this app, we allow it to proceed (OCR will just fail gracefully later).


def extract_text_from_frame(frame: np.ndarray) -> str:
    """
    Extracts text from a single video frame using EasyOCR.

    Args:
        frame (np.ndarray): The image frame in OpenCV format (BGR numpy array).

    Returns:
        str: A single string containing all detected text elements joined by spaces.
             Returns an empty string if processing fails or no text is found.
    """
    if frame is None:
        return ""

    try:
        # EasyOCR accepts images in BGR (OpenCV default) or RGB.
        # Setting detail=0 returns a simple list of detected text strings,
        # ignoring bounding box coordinates and confidence scores.
        result = reader.readtext(frame, detail=0)

        # Join the list of detected strings into a single searchable text block
        text = " ".join(result)
        return text.strip()

    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return ""


if __name__ == "__main__":
    # Test block to verify OCR functionality independently
    
    # Define potential paths to find the video data directory
    # depending on where this script is executed from.
    possible_paths = [
        "data/videos",
        "../../data/videos",
        "videos"
    ]
    
    video_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            video_dir = path
            break
            
    if video_dir:
        # Find the first available .mp4 file for testing
        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        
        if video_files:
            video_path = os.path.join(video_dir, video_files[0])
            print(f"Testing OCR on video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            
            # Jump to 10 seconds (10000 ms) into the video
            # News videos often have tickers/banners appear after the intro
            cap.set(cv2.CAP_PROP_POS_MSEC, 10000)
            success, frame = cap.read()
            
            if success:
                print("Frame extracted. Running OCR...")
                text_result = extract_text_from_frame(frame)
                print(f"Detected Text: '{text_result}'")
            else:
                print("Failed to extract frame from video.")
                
            cap.release()
        else:
            print(f"No .mp4 files found in {video_dir}")
    else:
        print("Could not find video directory for testing.")