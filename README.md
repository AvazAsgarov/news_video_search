Semantic News Video Search 📺🔍

A Multimodal RAG (Retrieval-Augmented Generation) system that makes news video archives searchable via natural language. By analyzing Audio, Visuals, and On-Screen Text, users can find precise video segments and receive AI-generated answers to their queries.

🚀 Features

Multimodal Analysis: Combines three data streams to understand video content:

Audio: Speech-to-Text transcription (OpenAI Whisper).

Visual: Scene description and context (GPT-4o Vision).

Text (OCR): Extraction of tickers, banners, and chyrons (EasyOCR).

Semantic Search: Uses vector embeddings (ChromaDB) to find videos by meaning, not just keywords.

Smart Segmentation: Splits videos into 20s sliding windows to preserve context.

Metadata Enrichment:

NER: Identifies People, Organizations, and Locations (Spacy).

Auto-Tagging: Categorizes videos into topics like Politics, Sports, or Economy.

RAG QA: Generates factual, context-aware answers to user questions based on the retrieved video content.

📂 Project Structure

news_video_search/
├── app/                        # Main backend logic
│   ├── config.py               # Configuration & Path settings
│   ├── process_videos.py       # Master ingestion pipeline (Run this first!)
│   ├── rag_search.py           # RAG generation logic
│   ├── services/               # External API integrations
│   │   ├── audio_service.py    # Whisper transcription
│   │   ├── vision_service.py   # GPT-4o visual analysis
│   │   └── embedding_service.py# Vector DB interface
│   └── core/                   # Core processing algorithms
│       ├── video_processor.py  # Segmentation logic
│       ├── ner_analyzer.py     # Named Entity Recognition
│       ├── ocr_processor.py    # Text extraction
│       └── tag_generator.py    # Automatic topic labeling
├── data/                       # Data storage (Created automatically)
│   ├── videos/                 # Put your .mp4 files here
│   ├── vector_db/              # ChromaDB storage
│   └── generated_tags.json     # Taxonomy tags
├── frontend/
│   └── streamlit_app.py        # Web Interface
├── requirements.txt            # Python dependencies
└── .env                        # API Keys (Create this file)


🛠️ Installation

1. Clone the Repository

git clone [https://github.com/yourusername/news-video-search.git](https://github.com/yourusername/news-video-search.git)
cd news_video_search


2. Create a Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Download Language Models

You need to download the Spacy English model for NER:

python -m spacy download en_core_web_sm


5. Configure API Keys

Create a .env file in the root directory and add your OpenAI API Key:

OPENAI_API_KEY=sk-proj-your-api-key-here


🏃 Usage

Step 1: Add Videos

Place your news video files (.mp4 format) into the data/videos/ folder.
If the folder doesn't exist, the script will create it, or you can create it manually.

Step 2: Process the Videos (Ingestion)

Run the master pipeline. This will transcribe, analyze, and index your videos into the vector database.

python -m app.process_videos


Note: This process may take a few minutes depending on the length of your videos.

Step 3: Generate Tags (Optional)

To create category labels (Politics, Sports, etc.) for the UI:

python -m app.core.tag_generator


Step 4: Launch the Search App

Start the Streamlit frontend:

streamlit run frontend/streamlit_app.py


Open your browser at http://localhost:8501.

🧪 Example Queries

Try asking questions like:

"What is the update on the peace talks?" (Tests OCR & Audio)

"Show me the interview with Ronaldo." (Tests Visual Recognition)

"How much aid did Zelensky ask for?" (Tests Factual Retrieval)

"Why was the event in India cancelled?" (Tests Complex Reasoning)

🧠 Technical Details

Segmentation: Videos are sliced into 20-second chunks with a 10-second overlap. This ensures that sentences or events occurring at the cut point are captured fully in the subsequent chunk.

Visual Optimization: To save costs, the system calculates the Mean Squared Error (MSE) between frames. It only sends a frame to GPT-4o if the scene has changed significantly.

Vector DB: We use ChromaDB with text-embedding-3-small for efficient local vector storage.

⚠️ Requirements

Python 3.9+

FFmpeg (Installed via moviepy, but ensure it's available on your system if issues arise)

OpenAI API Credit (GPT-4o and Whisper are paid APIs)