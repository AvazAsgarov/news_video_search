"""
Core module for video ingestion and temporal segmentation.

This module provides utilities to scan the data directory for video files,
extract metadata using OpenCV, and generate sliding window intervals for
subsequent processing steps.
"""

import os
import sys
import uuid
from glob import glob
from typing import List, Dict, Optional, Any

import cv2

# Ensure project root is in sys.path for standalone execution
try:
    from app.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.config import Config


def get_video_duration(video_path: str) -> Optional[float]:
    """
    Extracts the total duration of a video file in seconds.

    Args:
        video_path (str): The absolute or relative path to the video file.

    Returns:
        Optional[float]: The duration in seconds. Returns None if the file 
                         cannot be opened or the frame rate is invalid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0.0

    cap.release()
    return duration


def ingest_videos(video_folder: str) -> List[Dict[str, Any]]:
    """
    Scans the specified folder for MP4 files and compiles metadata.

    Args:
        video_folder (str): The path to the directory containing video files.

    Returns:
        List[Dict[str, Any]]: A list of metadata dictionaries for each video,
                              including ID, filename, path, and duration.
    """
    video_files = glob(os.path.join(video_folder, "*.mp4"))
    ingested_videos = []

    print(f"Scanning directory: {video_folder}")

    for file_path in video_files:
        filename = os.path.basename(file_path)
        duration = get_video_duration(file_path)

        if duration is None:
            print(f"Error: Could not process {filename}. Skipping.")
            continue

        # Generate a unique short identifier for this video
        video_id = str(uuid.uuid4())[:8]

        video_metadata = {
            "video_id": video_id,
            "filename": filename,
            "file_path": file_path,
            "duration_seconds": round(duration, 2)
        }

        ingested_videos.append(video_metadata)
        print(f"Ingested: {filename} (ID: {video_id}) - {round(duration, 2)}s")

    return ingested_videos


def create_sliding_windows(
    video_duration: float, 
    window_size: int = 20, 
    step_size: int = 10
) -> List[Dict[str, float]]:
    """
    Creates overlapping time intervals for a given video duration.

    This ensures that context is preserved at the boundaries of each segment
    by overlapping the windows.

    Args:
        video_duration (float): The total length of the video in seconds.
        window_size (int): The duration of each segment in seconds.
        step_size (int): The number of seconds to advance for the next window.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing 'start' and 'end'
                                timestamps for each window.
    """
    windows = []
    current_start = 0.0

    while current_start < video_duration:
        current_end = current_start + window_size

        # Ensure the window does not exceed the total video duration
        if current_end > video_duration:
            current_end = video_duration

        # Skip segments that are too short (less than 1 second)
        if (current_end - current_start) < 1.0:
            break

        windows.append({
            "start": round(current_start, 2),
            "end": round(current_end, 2)
        })

        # Exit loop if the end of the video has been reached
        if current_end == video_duration:
            break

        current_start += step_size

    return windows


if __name__ == "__main__":
    # Test block to verify ingestion and window creation logic
    videos = ingest_videos(Config.VIDEO_DIR)
    
    if videos:
        first_video = videos[0]
        test_windows = create_sliding_windows(first_video['duration_seconds'])
        
        print(f"\nTest Summary for: {first_video['filename']}")
        print(f"Total duration: {first_video['duration_seconds']}s")
        print(f"Generated windows: {len(test_windows)}")
        print(f"Sample window: {test_windows[0]}")
    else:
        print("No videos found in the specified directory.")