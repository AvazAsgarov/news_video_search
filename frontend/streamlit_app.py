"""
Streamlit frontend application for News Video Search.

This module provides a web interface for searching video segments using natural
language queries, integrating retrieval logic with RAG generation service.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

# Add project root to system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.config import Config
from app.rag_search import generate_answer


class VideoSearchApp:
    """Main application class for video search functionality."""
    
    def __init__(self):
        """Initialize application components."""
        self.tags_data = self._load_tags()
        self.collection = self._initialize_chromadb()
        
    def _load_tags(self) -> Dict:
        """Load auto-generated tags from JSON file.
        
        Returns:
            Dictionary containing video tags data.
        """
        tags_data = {}
        if os.path.exists(Config.TAGS_FILE_PATH):
            try:
                with open(Config.TAGS_FILE_PATH, "r") as f:
                    tags_data = json.load(f)
            except Exception as e:
                st.error(f"Error loading tags file: {e}")
        return tags_data
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection.
        
        Returns:
            ChromaDB collection object for querying.
        """
        try:
            client = chromadb.PersistentClient(path=Config.CHROMA_DB_DIR)
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=Config.OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )
            return client.get_collection("news_videos", embedding_function=openai_ef)
        except Exception as e:
            st.error(f"Failed to initialize ChromaDB: {e}")
            return None
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds into MM:SS string.
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            Formatted time string (e.g., "01:30").
        """
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02d}:{int(seconds):02d}"
    
    @staticmethod
    def parse_context(context_text: str) -> Dict[str, str]:
        """Parse the multimodal context text into separate components.
        
        Args:
            context_text: The combined context string.
            
        Returns:
            Dictionary with parsed components (visual, ocr, audio).
        """
        parsed = {"visual": "", "ocr": "", "audio": ""}
        
        # Find each component using markers (adjust based on your actual format)
        if "[Visual Scene]:" in context_text and "[On-Screen Text]:" in context_text and "[Audio Transcript]:" in context_text:
            # Split by known markers
            try:
                # Find indices
                visual_start = context_text.find("[Visual Scene]:") + len("[Visual Scene]:")
                ocr_start = context_text.find("[On-Screen Text]:")
                audio_start = context_text.find("[Audio Transcript]:")
                
                # Extract each part
                parsed["visual"] = context_text[visual_start:ocr_start].strip()
                parsed["ocr"] = context_text[ocr_start + len("[On-Screen Text]:"):audio_start].strip()
                parsed["audio"] = context_text[audio_start + len("[Audio Transcript]:"):].strip()
            except Exception:
                # Fallback: return as is
                parsed["audio"] = context_text
        else:
            # Format not recognized, return as audio
            parsed["audio"] = context_text
        
        return parsed
    
    def search_videos(self, query_text: str, n_results: int = 3) -> Optional[Dict]:
        """Query ChromaDB collection for relevant video segments.
        
        Args:
            query_text: User's search query.
            n_results: Number of top results to return.
            
        Returns:
            Query results containing IDs, documents, and metadata.
        """
        if not self.collection:
            return None
            
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            st.error(f"Search failed: {e}")
            return None
    
    def get_video_metadata(self, video_filename: str) -> str:
        """Extract tags from tags data for a specific video.
        
        Args:
            video_filename: Name of the video file.
            
        Returns:
            Tags for the video.
        """
        return self.tags_data.get(video_filename, "General")
    
    def display_search_results(self, query: str, results: Dict) -> None:
        """Display search results in the Streamlit interface.
        
        Args:
            query: Original search query.
            results: Search results from ChromaDB.
        """
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Display AI summary
        st.markdown("### 🤖 AI Summary")
        with st.spinner("Generating AI summary..."):
            ai_summary = generate_answer(query, documents)
        
        # Display AI answer in a styled container
        with st.container():
            st.markdown(ai_summary)
            st.divider()
        
        # Results header
        st.subheader(f"📹 Found {len(ids)} Relevant Segments")
        
        # Display each video result
        for idx in range(len(ids)):
            self._display_video_result(
                idx=idx,
                metadata=metadatas[idx],
                document=documents[idx],
                video_count=len(ids)
            )
    
    def _display_video_result(self, idx: int, metadata: Dict, 
                            document: str, video_count: int) -> None:
        """Display individual video result.
        
        Args:
            idx: Result index.
            metadata: Video metadata.
            document: Text context/document.
            video_count: Total number of results.
        """
        video_filename = metadata['filename']
        start_time = metadata['start_time']
        end_time = metadata['end_time']
        
        # Get tags from loaded data
        tags = self.get_video_metadata(video_filename)
        
        # Extract named entities
        people = metadata.get('people', '')
        locations = metadata.get('locations', '')
        organizations = metadata.get('organizations', '')
        
        # Construct video path
        video_file_path = os.path.join(Config.VIDEO_DIR, video_filename)
        time_range = f"{self.format_timestamp(start_time)} - {self.format_timestamp(end_time)}"
        
        # Create two-column layout
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            self._display_video_player(video_file_path, start_time)
        
        with col2:
            self._display_metadata(
                video_filename=video_filename,
                tags=tags,
                time_range=time_range,
                people=people,
                locations=locations,
                organizations=organizations
            )
            
            # Context expander with improved formatting
            self._display_context_expander(document, idx)
        
        # Add divider between results
        if idx < video_count - 1:
            st.divider()
    
    def _display_video_player(self, video_path: str, start_time: float) -> None:
        """Display video player component.
        
        Args:
            video_path: Path to video file.
            start_time: Start time for video playback.
        """
        if os.path.exists(video_path):
            # Create a container for the video
            with st.container():
                st.video(video_path, start_time=int(start_time))
        else:
            st.error(f"Video file not found: {os.path.basename(video_path)}")
    
    def _display_metadata(self, video_filename: str, tags: str, time_range: str,
                         people: str, locations: str, organizations: str) -> None:
        """Display video metadata.
        
        Args:
            video_filename: Name of the video file.
            tags: Video tags.
            time_range: Formatted time range.
            people: Named entities - people.
            locations: Named entities - locations.
            organizations: Named entities - organizations.
        """
        # Create a styled metadata container
        with st.container():
            st.markdown(f"**🎬 Source:** `{video_filename}`")
            st.markdown(f"**🏷️ Topic:** {tags}")
            st.markdown(f"**⏱️ Time:** `{time_range}`")
            
            # Display named entities in a compact format
            if people or locations or organizations:
                st.markdown("**🔍 Named Entities**")
                if people:
                    st.markdown(f"👤 *People:* {people}")
                if locations:
                    st.markdown(f"📍 *Locations:* {locations}")
                if organizations:
                    st.markdown(f"🏢 *Organizations:* {organizations}")
    
    def _display_context_expander(self, document: str, idx: int) -> None:
        """Display parsed context in an expander.
        
        Args:
            document: Combined context document.
            idx: Index for unique keys.
        """
        with st.expander("📋 View AI Analysis", expanded=False):
            # Parse the multimodal context
            parsed_context = self.parse_context(document)
            
            # Display each component in separate sections
            if parsed_context.get("visual"):
                st.markdown("**🖼️ Visual Scene Analysis**")
                st.info(parsed_context["visual"])
            
            if parsed_context.get("ocr"):
                st.markdown("**📝 On-Screen Text (OCR)**")
                st.success(parsed_context["ocr"])
            
            if parsed_context.get("audio"):
                st.markdown("**🎙️ Audio Transcript**")
                st.warning(parsed_context["audio"])
            
            # If parsing failed, show raw document
            if not any(parsed_context.values()) and document:
                st.markdown("**📄 Raw Context**")
                st.text_area("Context", document, height=150, key=f"raw_context_{idx}", disabled=True)


def display_sidebar() -> None:
    """Display application sidebar with project information."""
    with st.sidebar:
        # Logo/Title
        st.markdown("## 📹 News Video Search")
        st.divider()
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system enables semantic search across news videos by analyzing:
        - Visual scenes with GPT-4o Vision
        - Audio transcripts with Whisper
        - On-screen text with EasyOCR
        """)
        
        st.divider()
        
        # How it works
        st.markdown("### ⚙️ How It Works")
        with st.expander("View Pipeline"):
            st.markdown("""
            **1. Ingest**: Videos are uploaded and stored locally
            
            **2. Process**: Each 20s chunk is analyzed:
               - 🔊 Audio: Whisper transcription
               - 🖼️ Visual: Scene understanding
               - 📝 Text: OCR extraction
            
            **3. Index**: Metadata is stored in ChromaDB
            
            **4. Retrieve**: Semantic search with RAG
            """)
        
        # Technical Stack
        st.markdown("### 🛠️ Technical Stack")
        st.markdown("""
        - **Audio**: OpenAI Whisper
        - **Visual**: GPT-4o Vision
        - **OCR**: EasyOCR
        - **Vector DB**: ChromaDB
        - **LLM**: GPT-4o
        """)
        
        # Video information
        st.divider()
        st.markdown("### 📊 Video Info")
        # This could be dynamic based on available videos
        st.caption("Videos are processed in 20-second overlapping segments")


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Semantic News Video Search",
        layout="wide",
        page_icon="📹",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .stTextInput input {
        border-radius: 10px;
    }
    .stButton button {
        border-radius: 10px;
        border: 1px solid #4CAF50;
    }
    .video-container {
        border-radius: 10px;
        padding: 10px;
        background-color: #f0f2f6;
    }
    .metadata-box {
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize application
    app = VideoSearchApp()
    
    # Header section with better layout
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown("<h1 class='main-header'>🔍 Semantic News Video Search</h1>", unsafe_allow_html=True)
        st.markdown("""
        Search inside news videos using natural language. Find relevant segments based on:
        - What you **see** (visual scenes)
        - What you **hear** (transcribed dialogue)
        - What you **read** (on-screen text)
        """)
    
    with col2:
        st.markdown("")
        # Add refresh/status button if needed
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Search input with improved placeholder
    query = st.text_input(
        "**Search Query**",
        placeholder="Example: 'Show me scenes with politicians discussing peace talks'",
        help="Ask a question or describe what you're looking for in the videos",
        key="search_input"
    )
    
    # Process search query
    if query:
        st.divider()
        
        with st.spinner("🔍 Searching video database..."):
            results = app.search_videos(query)
            
            if results and results['documents']:
                app.display_search_results(query, results)
            else:
                st.warning("No relevant video segments found. Try a different query or check if videos have been processed.")
                st.markdown("**💡 Tips:**")
                st.markdown("""
                - Be specific about people, locations, or events
                - Try different wording for the same concept
                - Use natural language as you would ask a colleague
                """)
    
    # Display examples when no query is entered
    else:
        st.divider()
        st.markdown("### 💡 Example Queries")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown("**👥 People & Events**")
                st.caption("'Find scenes with presidents shaking hands'")
                st.caption("'Show me interviews with the Prime Minister'")
        
        with col2:
            with st.container():
                st.markdown("**📍 Locations**")
                st.caption("'Find footage from Ukraine peace talks'")
                st.caption("'Show me scenes in Washington D.C.'")
        
        with col3:
            with st.container():
                st.markdown("**📰 Topics**")
                st.caption("'Find economic discussions'")
                st.caption("'Show me sports celebrations'")
    
    # Display sidebar
    display_sidebar()


if __name__ == "__main__":
    main()