"""
Configuration module for the News Video Search application.

This module loads environment variables and defines file paths used throughout the
application, ensuring consistent access to API keys and directory locations regardless
of the execution context.
"""

import os
import warnings
from dotenv import load_dotenv

# Load environment variables from a .env file or system environment
load_dotenv()


class Config:
    """
    Central configuration class for application settings.
    Defines absolute paths for data storage and retrieves API keys.
    """

    # Retrieve OpenAI API Key from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Calculate the Base Directory (Project Root)
    # Assumes this file is located at: project_root/app/config.py
    # 1. os.path.abspath(__file__) -> project_root/app/config.py
    # 2. os.path.dirname(...)      -> project_root/app
    # 3. os.path.dirname(...)      -> project_root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the primary data directory
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Define specific subdirectories for storage
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    CHROMA_DB_DIR = os.path.join(DATA_DIR, "vector_db")
    TEMP_AUDIO_DIR = os.path.join(DATA_DIR, "temp_audio")
    TAGS_FILE_PATH = os.path.join(DATA_DIR, "generated_tags.json")

    # Validate critical configuration
    if not OPENAI_API_KEY:
        warnings.warn(
            "OPENAI_API_KEY is not set. Please set it in your .env file or environment variables.",
            UserWarning
        )

    # Ensure critical directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)