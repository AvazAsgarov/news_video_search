"""
Master pipeline script for video processing and indexing.

This script orchestrates the entire ingestion flow:
1. Ingests video metadata.
2. Transcribes audio tracks.
3. Segments videos into sliding windows.
4. Generates visual captions using GPT-4o.
5. Extracts on-screen text using OCR.
6. Analyzes entities (NER) in the transcript.
7. Indexes all data into ChromaDB for search.
"""

import os
import sys

# Ensure project root is in sys.path for standalone execution
try:
    from app.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.config import Config

# Import Core Modules
from app.core.video_processor import ingest_videos, create_sliding_windows
from app.core.ner_analyzer import extract_entities
from app.core.ocr_processor import extract_text_from_frame

# Import Service Modules
from app.services.audio_service import extract_audio, transcribe_audio
from app.services.vision_service import (
    extract_frame_at_time,
    get_frame_difference,
    generate_visual_caption
)
from app.services.embedding_service import add_chunk_to_db


def process_single_video(video_meta: dict) -> None:
    """
    Runs the full processing pipeline on a single video file.

    Args:
        video_meta (dict): Metadata dictionary containing file path, ID, and duration.
    """
    video_path = video_meta['file_path']
    video_id = video_meta['video_id']
    duration = video_meta['duration_seconds']
    filename = video_meta['filename']

    print(f"\nPROCESSING: {filename} (ID: {video_id})")

    # ---------------------------------------------------------
    # Step 1: Audio Processing (Transcription)
    # ---------------------------------------------------------
    audio_path = extract_audio(video_path)
    if not audio_path:
        print(f"Skipping {filename}: Audio extraction failed.")
        return

    segments = transcribe_audio(audio_path)
    
    # Clean up temporary audio file to save space
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except PermissionError:
            print("Warning: Could not delete temp audio file (file in use).")

    if not segments:
        print(f"Skipping {filename}: Transcription returned no data.")
        return

    # ---------------------------------------------------------
    # Step 2: visual Processing Setup
    # ---------------------------------------------------------
    previous_frame = None
    previous_caption = "No visual context available."
    scene_change_threshold = 50.0  # MSE threshold for detecting new scenes

    # ---------------------------------------------------------
    # Step 3: Sliding Window Segmentation
    # ---------------------------------------------------------
    # We slice the video into overlapping 20-second chunks
    chunks = create_sliding_windows(duration, window_size=20, step_size=10)
    print(f"Splitting into {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        start_t = chunk['start']
        end_t = chunk['end']

        # -----------------------------------------------------
        # A. Audio Context
        # -----------------------------------------------------
        # Filter transcript segments that fall within this time window
        chunk_text_parts = []
        for seg in segments:
            # Check for temporal overlap
            if seg.start < end_t and seg.end > start_t:
                chunk_text_parts.append(seg.text)
        
        audio_text = " ".join(chunk_text_parts).strip()

        # -----------------------------------------------------
        # B. Visual Context (Vision + OCR)
        # -----------------------------------------------------
        # Extract a keyframe from the middle of the chunk
        midpoint = (start_t + end_t) / 2
        current_frame = extract_frame_at_time(video_path, midpoint)

        visual_caption = ""
        ocr_text = ""

        if current_frame is not None:
            # 1. Visual Captioning (Scene Description)
            # Calculate difference from the previous processed frame
            diff = get_frame_difference(previous_frame, current_frame)

            # Optimization: Only call GPT-4o if the scene has changed significantly
            # Always process the first chunk (i == 0)
            if i == 0 or diff > scene_change_threshold:
                visual_caption = generate_visual_caption(current_frame)
                previous_caption = visual_caption
                previous_frame = current_frame
            else:
                # Reuse the previous caption to save API costs and time
                visual_caption = previous_caption
                # Update reference frame to track gradual changes
                previous_frame = current_frame

            # 2. OCR Extraction (Text on Screen)
            # We run OCR on every keyframe to catch fast-moving tickers/headlines
            ocr_text = extract_text_from_frame(current_frame)
            if len(ocr_text) > 5:
                print(f"   OCR Detected: {ocr_text[:50]}...")

        # -----------------------------------------------------
        # C. Entity Analysis (NER)
        # -----------------------------------------------------
        # Extract people, locations, and organizations from the audio transcript
        entities = extract_entities(audio_text)
        people_str = ", ".join(entities['PERSON'])
        orgs_str = ", ".join(entities['ORG'])
        locs_str = ", ".join(entities['GPE'])

        # -----------------------------------------------------
        # D. Data Synthesis & Indexing
        # -----------------------------------------------------
        # Construct the final semantic text block for embedding
        final_text = (
            f"[Visual Scene]: {visual_caption} "
            f"[On-Screen Text]: {ocr_text} "
            f"[Audio Transcript]: {audio_text}"
        )

        # Save to ChromaDB
        add_chunk_to_db(
            video_id=video_id,
            start_time=start_t,
            end_time=end_t,
            text=final_text,
            metadata={
                "filename": filename,
                "people": people_str,
                "organizations": orgs_str,
                "locations": locs_str
            }
        )

        # Logging progress
        if people_str or locs_str:
            print(f"   Saved Chunk {i+1}/{len(chunks)} | Entities: {people_str} - {locs_str}")
        else:
            print(f"   Saved Chunk {i+1}/{len(chunks)}")


def main():
    """
    Main entry point for the batch processing script.
    """
    # 1. Scan for videos
    videos = ingest_videos(Config.VIDEO_DIR)
    
    if not videos:
        print(f"No videos found in {Config.VIDEO_DIR}. Please add .mp4 files.")
        return

    print(f"Found {len(videos)} videos to process.")

    # 2. Process each video
    for video in videos:
        try:
            process_single_video(video)
        except Exception as e:
            print(f"Critical error processing video {video['filename']}: {e}")

    print("\nAll videos processed and stored in Vector DB.")


if __name__ == "__main__":
    main()