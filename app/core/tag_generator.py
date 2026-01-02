"""
Core module for automatic video tagging.

This module analyzes video transcripts or summaries using GPT-4o to assign
consistent topic labels (taxonomy) to each video, facilitating categorized search.
"""

import json
import os
import sys
from typing import Dict

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Ensure imports work from project root or direct execution
try:
    from app.config import Config
    from app.core.video_processor import ingest_videos
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.config import Config
    from app.core.video_processor import ingest_videos

# Initialize OpenAI Client
client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Define a consistent list of tags (Taxonomy)
TAXONOMY = [
    "Politics",
    "Conflict/War",
    "Sports",
    "Economy",
    "Technology",
    "Weather",
    "Health",
    "Entertainment"
]


def classify_video_content(text_summary: str) -> str:
    """
    Uses GPT to categorize the text into exactly 1 or 2 tags from the defined taxonomy.

    Args:
        text_summary (str): A summary or snippet of the video content.

    Returns:
        str: Comma-separated tags (e.g., "Politics, Economy") or "General" if no match.
    """
    prompt = f"""
    You are an auto-tagging system for a news archive.
    
    Allowed Tags: {", ".join(TAXONOMY)}
    
    Task: Logically assign the most relevant 1 or 2 tags to the following text.
    Rules:
    1. ONLY use tags from the Allowed Tags list.
    2. Return them as a comma-separated string (e.g., "Politics, Economy").
    3. If nothing matches, return "General".
    
    Text to classify:
    "{text_summary[:1000]}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        return content if content else "General"
    except Exception as e:
        print(f"Error classifying content: {e}")
        return "General"


def generate_video_tags() -> None:
    """
    Scans the vector database for processed video chunks, classifies their content,
    and saves the tags to a JSON file for the frontend to use.
    """
    # Initialize dictionary to store tags
    video_tags: Dict[str, str] = {}

    # Initialize ChromaDB connection
    # os.makedirs is not needed here as we expect the DB to already exist
    db_client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=Config.OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    
    try:
        collection = db_client.get_collection("news_videos", embedding_function=openai_ef)
    except Exception as e:
        print(f"Error accessing collection 'news_videos': {e}")
        print("Please ensure you have run 'process_videos.py' to populate the database.")
        return

    # Get list of videos to process
    videos = ingest_videos(Config.VIDEO_DIR)

    print("\nStarting automatic tag generation...")

    for video in videos:
        vid_filename = video['filename']
        print(f"Analyzing content for: {vid_filename}")

        # Query ChromaDB for this video's content
        # We limit to 3 chunks (~1 minute) which is sufficient for high-level classification
        results = collection.get(
            where={"filename": vid_filename},
            limit=3,
            include=["documents"]
        )

        if results and results['documents']:
            # Combine the text from the retrieved chunks into one context block
            combined_text = " ".join(results['documents'])

            # Generate tags using LLM
            tags = classify_video_content(combined_text)
            video_tags[vid_filename] = tags
            print(f"   -> Assigned Tags: [{tags}]")
        else:
            print("   -> No data found in DB. Tagging as Uncategorized.")
            video_tags[vid_filename] = "Uncategorized"

    # Save tags to a local JSON file in the DATA directory
    # The frontend reads this file to display tags instantly without re-querying the API
    output_path = Config.TAGS_FILE_PATH
    
    try:
        with open(output_path, "w") as f:
            json.dump(video_tags, f, indent=4)
        print(f"\nTags successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving tags file: {e}")


if __name__ == "__main__":
    generate_video_tags()