"""
Service module for vector database interactions.

This module manages the connection to ChromaDB, handles the embedding of text
using OpenAI's models, and provides functions to store and retrieve video segments.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

# Ensure project root is in sys.path for standalone execution
try:
    from app.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.config import Config

# Initialize ChromaDB Client
# Config.CHROMA_DB_DIR provides an absolute path, ensuring the DB is found 
# regardless of where the script is executed.
os.makedirs(Config.CHROMA_DB_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)

# Initialize OpenAI Embedding Function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=Config.OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# Get or Create the Collection (acts like a table in SQL)
collection = client.get_or_create_collection(
    name="news_videos",
    embedding_function=openai_ef
)


def add_chunk_to_db(
    video_id: str,
    start_time: float,
    end_time: float,
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Adds a single video chunk to the vector database with auto-retry logic.

    Args:
        video_id (str): Unique identifier for the video.
        start_time (float): Start time of the chunk in seconds.
        end_time (float): End time of the chunk in seconds.
        text (str): The combined text content (Audio + Visual + OCR).
        metadata (Optional[Dict[str, Any]]): Additional metadata (tags, entities, etc.).
    """
    if metadata is None:
        metadata = {}

    # Ensure mandatory metadata is present
    metadata["video_id"] = video_id
    metadata["start_time"] = start_time
    metadata["end_time"] = end_time

    # Create a unique ID for the chunk to prevent duplicates
    chunk_id = f"{video_id}_{start_time}_{end_time}"

    # Retry logic to handle potential DNS or connection glitches
    max_retries = 3
    for attempt in range(max_retries):
        try:
            collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[chunk_id]
            )
            # If successful, exit the retry loop
            break
        except Exception as e:
            print(f"Connection error saving chunk (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                print(f"Failed to save chunk {chunk_id} after {max_retries} attempts.")


def query_db(query_text: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Searches the database for the most relevant chunks based on semantic similarity.

    Args:
        query_text (str): The user's search query.
        n_results (int): Number of top results to return.

    Returns:
        Dict[str, Any]: The search results object from ChromaDB.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results


if __name__ == "__main__":
    print("Testing ChromaDB connection...")
    
    # Simple test to verify connection and retry logic
    try:
        add_chunk_to_db(
            video_id="test_service_connection",
            start_time=0.0,
            end_time=10.0,
            text="Testing connection from embedding service layer."
        )
        print("Connection successful.")
        
        # Test query
        print("Testing query...")
        test_results = query_db("Testing connection")
        if test_results['ids']:
            print("Query successful.")
        else:
            print("Query returned no results (expected if DB was empty).")
            
    except Exception as e:
        print(f"Service test failed: {e}")