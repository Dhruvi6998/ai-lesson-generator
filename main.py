# backend.py - FREE AI APIs Version (Groq + Gemini + HuggingFace)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import tempfile
import re
from PyPDF2 import PdfReader
import requests
import json

# API Models
class QuestionRequest(BaseModel):
    question: str
    api_provider: str = "groq"  # groq, gemini, huggingface
    api_key: str

class QuizRequest(BaseModel):
    topic: str
    api_provider: str = "groq"
    api_key: str

class QuizResponse(BaseModel):
    questions: List[Dict]

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    model_used: str

app = FastAPI(title="Free AI Lesson Generator", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Document Store
class DocumentStore:
    def __init__(self):
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = []
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
            print("✅ Using Sentence Transformers for better search")
        except:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,3), stop_words='english')
            self.vectors = None
            self.use_embeddings = False
            print("⚠️ Using TF-IDF (install sentence-transformers for better results)")
    
    def clear(self):
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = []
        if not self.use_embeddings:
            self.vectors = None

doc_store = DocumentStore()

# Document Processor with Topic Detection
class DocumentProcessor:
    @staticmethod
    def extract_text(file_path: str, file_type: str) -> str:
        try:
            if file_type == 'pdf':
                reader = PdfReader(file_path)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n{page_text}\n"
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    
    @staticmethod
    def detect_topics(text: str) -> List[Dict]:
        """Detect topics/sections in document"""
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Find headings - multiple patterns
        patterns = [
            r'\n([A-Z][A-Z\s]{8,80})\n',  # ALL CAPS
            r'\n(\d+\.?\s*[A-Z][^\n]{8,70})\n',  # Numbered sections
            r'\n([A-Z][^\n]{8,70}:)\n',  # Heading with colon
            r'(Chapter\s+\d+[^\n]{0,50})',  # Chapters
            r'(UNIT\s+\d+[^\n]{0,50})',  # Units
        ]
        
        sections = []
        found_headings = []
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                heading = match.group(1).strip()
                pos = match.start()
                found_headings.append((pos, heading))
        
        # Sort by position
        found_headings.sort()
        
        # Extract content between headings
        for i, (pos, heading) in enumerate(found_headings):
            start = pos + len(heading)
            end = found_headings[i+1][0] if i+1 < len(found_headings) else len(text)
            content = text[start:end].strip()
            
            if len(content.split()) > 50:
                sections.append({
                    'topic': heading[:80],
                    'content': content
                })
        
        # If no sections found, create artificial ones
        if not sections:
            words = text.split()
            chunk_size = 1000
            for i in range(0, len(words), chunk_size):
                sections.append({
                    'topic': f'Section {i//chunk_size + 1}',
                    'content': ' '.join(words[i:i+chunk_size])
                })
        
        return sections
    
    @staticmethod
    def create_chunks(sections: List[Dict]) -> List[Dict]:
        """Create focused chunks from sections"""
        chunks = []
        
        for section in sections:
            content = section['content']
            topic = section['topic']
            
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            current = []
            word_count = 0
            
            for sent in sentences:
                sent_words = len(sent.split())
                
                if word_count + sent_words > 300 and current:
                    chunks.append({
                        'text': ' '.join(current),
                        'topic': topic,
                        'words': word_count
                    })
                    # Keep last sentence for continuity
                    current = [current[-1], sent] if len(current) > 0 else [sent]
                    word_count = sum(len(s.split()) for s in current)
                else:
                    current.append(sent)
                    word_count += sent_words
            
            if current and word_count > 30:
                chunks.append({
                    'text': ' '.join(current),
                    'topic': topic,
                    'words': word_count
                })
        
        return chunks

# Smart Retriever
class SmartRetriever:
    @staticmethod
    def retrieve(query: str, top_k: int = 3) -> List[Dict]:
        if not doc_store.chunks:
            return []
        
        # Expand query
        queries = [query, query.lower()]
        words = [w for w in query.lower().split() if len(w) > 3]
        queries.extend(words)
        
        all_scores = {}
        
        for q in queries:
            if doc_store.use_embeddings:
                import numpy as np
                q_emb = doc_store.model.encode([q])[0]
                
                for i, emb in enumerate(doc_store.embeddings):
                    sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                    
                    # Boost if topic matches
                    topic = doc_store.chunk_metadata[i]['topic'].lower()
                    boost = 1.5 if any(word in topic for word in q.lower().split()) else 1.0
                    
                    score = sim * boost
                    if i in all_scores:
                        all_scores[i] = max(all_scores[i], score)
                    else:
                        all_scores[i] = score
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                q_vec = doc_store.vectorizer.transform([q])
                sims = cosine_similarity(q_vec, doc_store.vectors)[0]
                
                for i, sim in enumerate(sims):
                    topic = doc_store.chunk_metadata[i]['topic'].lower()
                    boost = 1.5 if any(word in topic for word in q.lower().split()) else 1.0
                    score = sim * boost
                    
                    if i in all_scores:
                        all_scores[i] = max(all_scores[i], score)
                    else:
                        all_scores[i] = score
        
        # Get top results
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        threshold = 0.25 if doc_store.use_embeddings else 0.1
        
        results = []
        for idx, score in sorted_items[:top_k]:
            if score > threshold:
                results.append({
                    'text': doc_store.chunks[idx],
                    'topic': doc_store.chunk_metadata[idx]['topic'],
                    'score': float(score)
                })
        
        return results

# FREE AI API Service
class FreeAIService:
    """Supports: Groq (BEST), Google Gemini, HuggingFace"""
    
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        self.api_key = api_key
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            if self.provider == "groq":
                return self._groq_generate(prompt, max_tokens)
            elif self.provider == "gemini":
                return self._gemini_generate(prompt, max_tokens)
            elif self.provider == "huggingface":
                return self._huggingface_generate(prompt, max_tokens)
            else:
                return "❌ Invalid API provider. Use: groq, gemini, or huggingface"
        except Exception as e:
            return f"⚠️ API Error: {str(e)}"
    
    def _groq_generate(self, prompt: str, max_tokens: int) -> str:
        """Groq API - FASTEST & BEST FREE API"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Try models in order of preference
        models = [
            "llama-3.3-70b-versatile",  # Latest Llama 3.3
            "llama-3.1-8b-instant",     # Fast fallback
            "mixtral-8x7b-32768"        # Alternative
        ]
        
        for model in models:
            try:
                data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful educational assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code != 400:  # If not model error, don't retry
                    raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            except Exception as e:
                if model == models[-1]:  # Last model, raise error
                    raise Exception(f"Groq API error: {str(e)}")
                continue  # Try next model
    
    def _gemini_generate(self, prompt: str, max_tokens: int) -> str:
        """Google Gemini API - FREE & GOOD"""
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
    
    def _huggingface_generate(self, prompt: str, max_tokens: int) -> str:
        """HuggingFace API - FREE"""
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                return result[0].get('generated_text', '').strip()
            return str(result)
        else:
            raise Exception(f"HuggingFace API error: {response.status_code}")

# Quiz Parser
class QuizParser:
    @staticmethod
    def parse(text: str) -> List[Dict]:
        questions = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        current_q = None
        
        for line in lines:
            # Question
            q_match = re.match(r'(?:Q\d+|Question\s*\d+|^\d+)[:\.\)]\s*(.+)', line, re.IGNORECASE)
            if q_match:
                if current_q and 'question' in current_q and 'answer' in current_q and len(current_q.get('options', {})) == 4:
                    questions.append(current_q)
                
                current_q = {
                    'question': q_match.group(1).strip(),
                    'options': {}
                }
                continue
            
            # Options
            opt_match = re.match(r'([A-D])[:\.\)]\s*(.+)', line, re.IGNORECASE)
            if opt_match and current_q:
                letter = opt_match.group(1).upper()
                current_q['options'][letter] = opt_match.group(2).strip()
                continue
            
            # Answer
            ans_match = re.match(r'(?:Answer|Correct|Key)[:\s]*([A-D])', line, re.IGNORECASE)
            if ans_match and current_q:
                current_q['answer'] = ans_match.group(1).upper()
        
        if current_q and 'question' in current_q and 'answer' in current_q and len(current_q.get('options', {})) == 4:
            questions.append(current_q)
        
        return questions[:5]

# API Endpoints
@app.get("/")
async def root():
    return {
        "name": "Free AI Lesson Generator",
        "version": "4.0",
        "supported_apis": {
            "groq": "⭐ RECOMMENDED - Fastest, best quality, free",
            "gemini": "✅ Google's AI, free tier, very good",
            "huggingface": "✅ Many models, free"
        },
        "get_api_keys": {
            "groq": "https://console.groq.com/keys (FREE)",
            "gemini": "https://makersuite.google.com/app/apikey (FREE)",
            "huggingface": "https://huggingface.co/settings/tokens (FREE)"
        }
    }

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        doc_store.clear()
        all_chunks = []
        all_metadata = []
        all_topics = []
        
        for file in files:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                file_type = 'pdf' if file.filename.endswith('.pdf') else 'txt'
                text = DocumentProcessor.extract_text(tmp_path, file_type)
                
                # Detect topics
                sections = DocumentProcessor.detect_topics(text)
                print(f"📚 Found {len(sections)} topics in {file.filename}")
                
                # Create chunks
                chunks = DocumentProcessor.create_chunks(sections)
                
                for chunk in chunks:
                    all_chunks.append(chunk['text'])
                    all_metadata.append({
                        'topic': chunk['topic'],
                        'words': chunk['words'],
                        'source': file.filename
                    })
                    if chunk['topic'] not in all_topics:
                        all_topics.append(chunk['topic'])
                
            finally:
                os.unlink(tmp_path)
        
        doc_store.chunks = all_chunks
        doc_store.chunk_metadata = all_metadata
        
        # Create embeddings
        if doc_store.use_embeddings:
            doc_store.embeddings = doc_store.model.encode(all_chunks)
        else:
            doc_store.vectors = doc_store.vectorizer.fit_transform(all_chunks)
        
        print(f"✅ Topics: {', '.join(all_topics[:5])}")
        
        return {
            "status": "success",
            "chunks_created": len(all_chunks),
            "topics_found": all_topics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        if not doc_store.chunks:
            raise HTTPException(status_code=400, detail="No documents uploaded")
        
        # Retrieve relevant chunks
        results = SmartRetriever.retrieve(request.question, top_k=2)
        
        if not results:
            available_topics = list(set([m['topic'] for m in doc_store.chunk_metadata]))[:5]
            return AnswerResponse(
                answer=f"❌ No relevant information found. Try asking about: {', '.join(available_topics)}",
                sources=[],
                model_used="none"
            )
        
        # Build focused context
        context_parts = [f"[{r['topic']}]\n{r['text']}" for r in results]
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an educational assistant. Answer this question using ONLY the context provided.

Context from documents:
{context}

Question: {request.question}

Instructions:
- Answer in 2-4 clear sentences
- Use ONLY information from the context above
- Be specific and accurate
- If the context doesn't fully answer the question, say so

Answer:"""
        
        # Generate with selected AI
        ai = FreeAIService(request.api_provider, request.api_key)
        answer = ai.generate(prompt, max_tokens=300)
        
        model_names = {
            "groq": "Groq Llama 3.1 70B",
            "gemini": "Google Gemini Pro",
            "huggingface": "Mistral 7B"
        }
        
        return AnswerResponse(
            answer=answer,
            sources=[r['text'][:250] + "..." for r in results],
            model_used=model_names.get(request.api_provider, request.api_provider)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        if not doc_store.chunks:
            raise HTTPException(status_code=400, detail="No documents uploaded")
        
        # Get topic-specific content
        results = SmartRetriever.retrieve(request.topic, top_k=3)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No content found about '{request.topic}'")
        
        context = "\n\n".join([f"[{r['topic']}]\n{r['text']}" for r in results])
        
        prompt = f"""Create 4 different multiple-choice questions about: {request.topic}

Content:
{context}

Create 4 DIFFERENT types of questions:
1. Definition/Concept question
2. Application/Example question
3. Comparison/Analysis question
4. Inference question

ALL questions must come from the content above.

Format EXACTLY like this:
Q1: [Your definition question]?
A) [Option]
B) [Option]
C) [Option]
D) [Option]
ANSWER: A

Q2: [Your application question]?
A) [Option]
B) [Option]
C) [Option]
D) [Option]
ANSWER: B

Q3: [Your comparison question]?
A) [Option]
B) [Option]
C) [Option]
D) [Option]
ANSWER: C

Q4: [Your inference question]?
A) [Option]
B) [Option]
C) [Option]
D) [Option]
ANSWER: D

Generate the quiz now:"""
        
        # Generate with AI
        ai = FreeAIService(request.api_provider, request.api_key)
        quiz_text = ai.generate(prompt, max_tokens=1000)
        
        # Parse
        questions = QuizParser.parse(quiz_text)
        
        if len(questions) < 2:
            raise HTTPException(status_code=500, detail="Failed to generate quiz. Check your API key.")
        
        return QuizResponse(questions=questions)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    topics = list(set([m['topic'] for m in doc_store.chunk_metadata])) if doc_store.chunk_metadata else []
    return {
        "status": "online",
        "chunks_indexed": len(doc_store.chunks),
        "topics": topics,
        "ready": len(doc_store.chunks) > 0
    }

@app.delete("/clear")
async def clear_documents():
    doc_store.clear()
    return {"status": "cleared"}

if __name__ == "__main__":
    print("\n🚀 Starting FREE AI Lesson Generator")
    print("✨ Supports: Groq (BEST) ⭐, Google Gemini, HuggingFace")
    print("🆓 All APIs are FREE!")
    print("\n📝 Get API Keys:")
    print("   Groq: https://console.groq.com/keys")
    print("   Gemini: https://makersuite.google.com/app/apikey")
    print("   HuggingFace: https://huggingface.co/settings/tokens")
    print("\n🌐 API running on: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)