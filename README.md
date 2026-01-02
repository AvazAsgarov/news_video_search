# 📺 Semantic News Video Search | Multimodal RAG System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![AI-Powered](https://img.shields.io/badge/AI--Powered-Multimodal%20RAG-red.svg)

An intelligent video retrieval system that transforms news archives into searchable knowledge bases using multimodal AI. Search through hours of video content using natural language and get precise answers with exact timestamps.

## 🚀 Core Capabilities

### 🔍 **Intelligent Video Search**
- **Natural Language Queries**: Ask questions like you would ask a colleague
- **Multimodal Understanding**: Simultaneously analyzes audio, visuals, and text
- **Semantic Retrieval**: Finds content by meaning, not just keywords
- **Exact Timestamping**: Returns precise video segments for playback

### 🧠 **Advanced AI Analysis**
| **Modality** | **Technology** | **What It Captures** |
|--------------|---------------|----------------------|
| 🔊 **Audio** | OpenAI Whisper | Transcribed dialogue, speaker identification |
| 🖼️ **Visual** | GPT-4o Vision | Scene descriptions, activities, objects |
| 📝 **Text** | EasyOCR | On-screen text, tickers, chyrons, banners |
| 🏷️ **Metadata** | SpaCy + GPT-4o | Named entities, topics, classifications |

### ⚡ **Smart Processing**
- **Sliding Window Segmentation**: 20-second chunks with 50% overlap
- **Scene Change Detection**: Optimizes API calls using MSE analysis
- **Parallel Processing**: Efficient handling of multiple modalities
- **Vector Embeddings**: Semantic storage with ChromaDB

## 📁 Project Architecture

```
news_video_search/
├── 📂 app/                           # Core backend logic
│   ├── config.py                     # Environment & configuration
│   ├── process_videos.py             # ⚡ Master pipeline (run this first)
│   ├── rag_search.py                 # RAG answer generation
│   ├── 📂 services/                  # External API integrations
│   │   ├── audio_service.py          # Whisper transcription
│   │   ├── vision_service.py         # GPT-4o visual analysis
│   │   └── embedding_service.py      # Vector embedding generation
│   └── 📂 core/                      # Processing algorithms
│       ├── video_processor.py        # Sliding window segmentation
│       ├── ner_analyzer.py           # Named Entity Recognition
│       ├── ocr_processor.py          # On-screen text extraction
│       └── tag_generator.py          # Automatic topic classification
├── 📂 data/                          # Data storage (auto-created)
│   ├── videos/                       # 🎬 Place .mp4 files here
│   ├── vector_db/                    # ChromaDB vector storage
│   └── generated_tags.json           # Auto-generated taxonomy tags
├── 📂 frontend/
│   └── streamlit_app.py              # 🌐 Web interface
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template
└── README.md                         # This file
```

## 🛠️ Quick Start Installation

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/news-video-search.git
cd news_video_search

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API key
# Add: OPENAI_API_KEY=sk-proj-your-api-key-here
```

### 3. Install Language Models

```bash
# Install SpaCy model for NER
python -m spacy download en_core_web_sm
```

## 🏃‍♂️ Usage Guide

### **Step 1: Prepare Videos**
Place your `.mp4` video files in the `data/videos/` directory:
```bash
# Create directory if needed
mkdir -p data/videos

# Add your news videos here
# Supported formats: .mp4, .mov, .avi
```

### **Step 2: Process Videos**
Run the ingestion pipeline (this may take time depending on video length):
```bash
python -m app.process_videos
```
✅ This automatically:
- Segments videos into 20-second chunks
- Transcribes audio with Whisper
- Analyzes visual scenes with GPT-4o Vision
- Extracts on-screen text with EasyOCR
- Stores embeddings in ChromaDB

### **Step 3: Generate Tags** (Optional)
Create topic classifications for better filtering:
```bash
python -m app.core.tag_generator
```

### **Step 4: Launch Web Interface**
Start the search application:
```bash
streamlit run frontend/streamlit_app.py
```
🌐 Open browser at: `http://localhost:8501`

## 🔍 Example Search Queries

### 🎯 **Topic-Based Searches**
```
"Show me segments about economic policies"
"Find climate change discussions"
"Show me sports highlights"
```

### 👥 **People & Events**
```
"Find interviews with the President"
"Show me when the peace treaty was signed"
"Find speeches by the Prime Minister"
```

### 📍 **Location-Specific**
```
"Show me footage from Ukraine"
"Find segments filmed in Washington D.C."
"Show me events in India"
```

### 🔎 **Complex Queries**
```
"What was discussed about the recent election results?"
"Show me the debate about healthcare reform"
"Find moments when the stock market was mentioned"
```

## 🧠 Technical Deep Dive

### **Video Processing Pipeline**
```python
# 1. Segmentation
Video → 20s chunks (50% overlap)

# 2. Multimodal Analysis
Audio → Whisper → Transcript
Visual → GPT-4o Vision → Scene description
Text → EasyOCR → On-screen text extraction

# 3. Metadata Enrichment
NER → People, Organizations, Locations
Tagging → Topic classification (Politics, Sports, etc.)

# 4. Vector Storage
Combined text → OpenAI embeddings → ChromaDB
```

### **RAG Retrieval Flow**
1. **Query Processing**: User question → vector embedding
2. **Semantic Search**: Find top 3 relevant video chunks
3. **Context Assembly**: Combine transcripts, descriptions, OCR
4. **Answer Generation**: GPT-4o generates response using retrieved context
5. **Result Delivery**: Answer + exact timestamps + source video

## ⚙️ Configuration Options

### **Chunking Parameters**
Modify in `app/core/video_processor.py`:
```python
WINDOW_SIZE = 20      # seconds per chunk
STEP_SIZE = 10        # seconds of overlap
MAX_CHUNKS = 100      # limit per video
```

### **Scene Detection Threshold**
Adjust in `app/services/vision_service.py`:
```python
MSE_THRESHOLD = 1000  # Lower = more sensitive to changes
```

### **Database Settings**
Configure in `app/config.py`:
```python
CHROMA_DB_DIR = "data/vector_db"
EMBEDDING_MODEL = "text-embedding-3-small"
```

## 📊 Performance Optimization

### **API Cost Management**
- **Scene Detection**: Only call Vision API when scenes change significantly
- **Batch Processing**: Process multiple videos sequentially
- **Local Models**: Option to replace Whisper with local installation

### **Processing Speed**
- **Parallel Processing**: Audio, visual, and text extraction can be parallelized
- **Caching**: Results cached to avoid reprocessing
- **Incremental Updates**: Only process new video segments

## 🤝 Contributing

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black app/ frontend/ tests/
```

### **Adding New Features**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgements

- **OpenAI** for Whisper and GPT-4o APIs
- **ChromaDB** for vector storage solutions
- **Streamlit** for the web framework
- **EasyOCR** for text extraction capabilities

---

<div align="center">
Made with ❤️ for journalists, researchers, and media professionals
</div>
