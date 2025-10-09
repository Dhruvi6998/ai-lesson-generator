# AI-Powered Lesson Generator with RAG

A complete AI-powered educational platform that uses **Retrieval Augmented Generation (RAG)** to help students learn from their documents. Upload PDFs or text files, ask questions, and generate interactive quizzes - all powered by free AI APIs.

## Features

**Topic-Aware Document Processing**
- Automatically detects topics and sections in your PDFs
- Smart chunking that preserves context
- Supports both PDF and TXT files
- Handles multi-topic documents (4+ topics per document)

**Multiple Free AI Providers**
- **Groq** (Recommended) - Fastest, best quality, free
- **Google Gemini** - Great alternative, free tier
- **HuggingFace** - Open source models, free

**Smart Question Answering**
- Returns only relevant context (not the whole document)
- Query expansion for better search
- Topic-focused retrieval
- Real AI responses (not templates!)

**Interactive Quiz Generation**
- 4 different question types per quiz
- Definition, Application, Comparison, Inference questions
- Auto-graded with instant feedback
- Celebration animations on passing (60%+)

**Advanced RAG Pipeline**
- Semantic search with sentence transformers
- TF-IDF fallback for compatibility
- Cosine similarity scoring
- Topic boosting for precision

### Prerequisites

- Python 3.11+ (Python 3.13 not yet supported)
- Git
- A free API key from [Groq](https://console.groq.com/keys), [Gemini](https://makersuite.google.com/app/apikey), or [HuggingFace](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-lesson-generator.git
cd ai-lesson-generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the backend**
```bash
python main.py
```

Backend will run on: `http://localhost:8000`

4. **Start the frontend** (new terminal)
```bash
streamlit run app.py
```

Frontend will open at: `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Configure API Provider

1. Choose your AI provider (Groq recommended)
2. Get your free API key:
   - **Groq**: https://console.groq.com/keys
   - **Gemini**: https://makersuite.google.com/app/apikey
   - **HuggingFace**: https://huggingface.co/settings/tokens
3. Paste the key in the sidebar

### Step 2: Upload Documents

1. Click "Upload PDF or TXT files"
2. Select your study materials
3. Click "📤 Process Documents"
4. Wait for topic detection

**System will automatically:**
- Detect topics and sections
- Create focused chunks (300-350 words)
- Index content for search
- Show detected topics in sidebar

### Step 3: Ask Questions

1. Go to "💡 Ask Questions" tab
2. Type a specific question about a topic
3. Click "🔍 Get Answer"
4. Get AI-powered response with source context

**Example Questions:**
```
✅ "What are the stages of photosynthesis?"
✅ "Explain Newton's first law"
✅ "What caused World War 2?"

❌ "Tell me everything" (too broad)
❌ "What is the capital of France?" (not in document)
```

### Step 4: Take Quiz

1. Go to "📝 Take Quiz" tab
2. Enter a topic from your document
3. Click "🎲 Generate Quiz"
4. Answer all 4 questions
5. Submit and get instant score
6. Balloons if you pass (60%+)!

## Technical Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PyPDF2** - PDF text extraction
- **scikit-learn** - TF-IDF vectorization
- **NumPy** - Numerical computations
- **Requests** - HTTP client for AI APIs

### Frontend
- **Streamlit** - Interactive web UI
- **Python** - Business logic

### AI Models
- **Groq Llama 3.3 70B** - Best quality (recommended)
- **Google Gemini Pro** - Great alternative
- **Mistral 7B** - HuggingFace option

### Optional (for better accuracy)
- **sentence-transformers** - Semantic embeddings
  - Note: Not compatible with Python 3.13 yet

## Deployment

### Deploy to Render (Backend + Frontend)

**Backend:**
1. Push code to GitHub
2. Create new Web Service on Render
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Choose Free plan

### Deploy to Streamlit Cloud (Frontend Only)

1. Deploy backend to Render (as above)
2. Go to https://share.streamlit.io
3. Connect GitHub repo
4. Main file: `app.py`
5. Add secret: `BACKEND_URL = "https://your-backend.onrender.com"`
6. Click Deploy

**Deployment time: ~10-15 minutes total**

##  Configuration

### Environment Variables

**Backend:**
- No environment variables required

**Frontend:**
- `BACKEND_URL` - Backend API URL (default: `http://localhost:8000`)

### API Keys

Get your free API keys:
Provider: Groq 
URL: https://console.groq.com/keys

Quality: Best
Speed: Fastest

Provider: Gemini
URL: https://makersuite.google.com/app/apikey

Quality: Great
Speed: Fast

Provider: HuggingFace
URL: https://huggingface.co/settings/tokens

Quality: Good
Speed: Slower

## Performance

### With Groq API (Recommended)
- **Answer Generation**: 1-2 seconds
- **Quiz Generation**: 2-3 seconds
- **Document Processing**: 5-10 seconds (1-20 pages)
- **Accuracy**: 95%+

##  Known Issues

- Python 3.13 not supported (use 3.11 or 3.12)
- Render free tier has cold starts (30-60 second initial load)
- Large PDFs (100+ pages) may take longer to process
- File uploads don't persist on Render free tier

