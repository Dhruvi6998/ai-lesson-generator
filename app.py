# frontend.py - Streamlit Frontend
import streamlit as st
import requests
import time
import os

# Backend API URL
API_BASE_URL = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL", "http://localhost:8000")


def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_documents(files, api_url):
    """Upload documents to backend"""
    files_data = []
    for file in files:
        files_data.append(('files', (file.name, file.getvalue(), file.type)))
    
    response = requests.post(f"{api_url}/upload", files=files_data)
    return response.json()

def ask_question(question, api_key, api_provider, api_url):
    """Ask question to backend"""
    payload = {
        "question": question,
        "api_key": api_key,
        "api_provider": api_provider
    }
    response = requests.post(f"{api_url}/ask", json=payload)
    return response.json()

def generate_quiz(topic, api_key, api_provider, api_url):
    """Generate quiz from backend"""
    payload = {
        "topic": topic,
        "api_key": api_key,
        "api_provider": api_provider
    }
    response = requests.post(f"{api_url}/quiz", json=payload)
    return response.json()

def main():
    st.set_page_config(
        page_title="AI Lesson Generator",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-score {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin: 20px 0;
        }
        .question-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 5px solid #1f77b4;
        }
        .correct {
            color: #28a745;
            font-weight: bold;
        }
        .incorrect {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 AI-Powered Lesson Generator with RAG")
    st.markdown("**Backend-Powered Learning System with Retrieval Augmented Generation**")
    
    # Check backend status
    if not check_backend():
        st.error("⚠️ Backend is not running! Please start the backend server first:")
        st.code("python backend.py", language="bash")
        st.stop()
    
    # Initialize session state
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None
    if 'docs_uploaded' not in st.session_state:
        st.session_state.docs_uploaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend status
        try:
            status = requests.get(f"{API_BASE_URL}/status").json()
            st.success("✅ Backend Connected")
            st.metric("Chunks Indexed", status.get('chunks_indexed', 0))
            
            # Show detected topics
            if status.get('topics'):
                with st.expander("📑 Detected Topics"):
                    for topic in status['topics']:
                        st.write(f"• {topic}")
            
            if status.get('ready', False):
                st.info("📄 Ready for questions!")
        except:
            st.warning("⚠️ Could not fetch status")
        
        st.markdown("---")
        
        # API Configuration
        st.header("🤖 AI Provider")
        
        api_provider = st.selectbox(
            "Choose AI Provider:",
            options=["groq", "gemini", "huggingface"],
            format_func=lambda x: {
                "groq": "⭐ Groq (FASTEST & BEST - Recommended)",
                "gemini": "✅ Google Gemini (Very Good)",
                "huggingface": "✅ HuggingFace (Good)"
            }[x]
        )
        
        # API key help links
        api_links = {
            "groq": "console.groq.com/keys",
            "gemini": "makersuite.google.com/app/apikey",
            "huggingface": "huggingface.co/settings/tokens"
        }
        
        api_key = st.text_input(
            f"{api_provider.upper()} API Key:",
            type="password",
            help=f"Get free key from: {api_links[api_provider]}"
        )
        
        if not api_key:
            st.warning("⚠️ API key required for AI responses!")
            st.info("👆 All APIs are FREE! Get your key from the link above.")
        
        st.markdown("---")
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", type="primary"):
                with st.spinner("Uploading and processing documents..."):
                    try:
                        result = upload_documents(uploaded_files, API_BASE_URL)
                        st.success(f"✅ Processed {len(uploaded_files)} documents")
                        st.info(f"📊 Created {result.get('chunks_created', 0)} chunks")
                        
                        # Show detected topics
                        if result.get('topics_found'):
                            st.write("**📑 Topics Found:**")
                            for topic in result['topics_found']:
                                st.write(f"• {topic}")
                        
                        st.session_state.docs_uploaded = True
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        if st.session_state.docs_uploaded:
            if st.button("🗑️ Clear All Documents"):
                requests.delete(f"{API_BASE_URL}/clear")
                st.session_state.docs_uploaded = False
                st.session_state.current_quiz = None
                st.rerun()
    
    # Main content
    if not st.session_state.docs_uploaded:
        st.info("👈 Please upload documents from the sidebar to get started!")
        
        st.markdown("### 🎯 Features:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📖 Document Processing**
            - PDF & TXT support
            - Automatic chunking
            - Vector indexing
            """)
        
        with col2:
            st.markdown("""
            **🤖 RAG System**
            - Context retrieval
            - AI-powered answers
            - Semantic search
            """)
        
        with col3:
            st.markdown("""
            **✅ Interactive Quiz**
            - Auto-generated questions
            - Real-time scoring
            - Celebration effects
            """)
        
        return
    
    # Tabs for main features
    tab1, tab2 = st.tabs(["💡 Ask Questions", "📝 Take Quiz"])
    
    # Question Answering Tab
    with tab1:
        st.header("Ask a Question")
        st.markdown("Ask anything about your uploaded documents!")
        
        question = st.text_area(
            "Your Question:",
            placeholder="e.g., What are the main concepts explained in the document?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_btn = st.button("🔍 Get Answer", type="primary")
        
        if ask_btn and question:
            if not api_key:
                st.error("⚠️ Please enter your API key in the sidebar!")
            else:
                with st.spinner("🤔 Analyzing documents and generating answer..."):
                    try:
                        result = ask_question(question, api_key, api_provider, API_BASE_URL)
                        
                        st.markdown("### 📖 Answer:")
                        st.markdown(f"<div class='question-box'>{result['answer']}</div>", 
                                   unsafe_allow_html=True)
                        
                        st.caption(f"🤖 Generated by: {result.get('model_used', 'AI')}")
                        
                        # Show retrieved context
                        if result.get('sources'):
                            with st.expander("🔍 View Retrieved Context Chunks"):
                                for i, chunk in enumerate(result['sources'], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text_area(
                                        f"chunk_{i}",
                                        chunk,
                                        height=150,
                                        label_visibility="collapsed"
                                    )
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif ask_btn:
            st.warning("⚠️ Please enter a question!")
    
    # Quiz Tab
    with tab2:
        st.header("Test Your Knowledge")
        st.markdown("Generate a quiz based on your documents!")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            quiz_topic = st.text_input(
                "Quiz Topic:",
                placeholder="e.g., main concepts, chapter 1, photosynthesis, etc."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_quiz_btn = st.button("🎲 Generate Quiz", type="primary")
        
        if gen_quiz_btn and quiz_topic:
            if not api_key:
                st.error("⚠️ Please enter your API key in the sidebar!")
            else:
                with st.spinner("🎯 Generating quiz questions using AI..."):
                    try:
                        result = generate_quiz(quiz_topic, api_key, api_provider, API_BASE_URL)
                        st.session_state.current_quiz = result['questions']
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        st.success("✅ Quiz generated successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display quiz
        if st.session_state.current_quiz:
            st.markdown("---")
            
            for i, q in enumerate(st.session_state.current_quiz):
                st.markdown(f"<div class='question-box'>", unsafe_allow_html=True)
                st.markdown(f"### Question {i+1}")
                st.markdown(f"**{q['question']}**")
                
                options = list(q['options'].items())
                options_text = [f"{k}) {v}" for k, v in options]
                
                selected = st.radio(
                    f"Select your answer:",
                    options=options_text,
                    key=f"quiz_q_{i}",
                    label_visibility="collapsed"
                )
                
                st.session_state.quiz_answers[i] = selected.split(')')[0] if selected else None
                st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submit_btn = st.button("✅ Submit Quiz", type="primary")
            
            with col2:
                new_quiz_btn = st.button("🔄 New Quiz")
            
            if submit_btn:
                st.session_state.quiz_submitted = True
                st.rerun()
            
            if new_quiz_btn:
                st.session_state.current_quiz = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.rerun()
            
            # Show results
            if st.session_state.quiz_submitted:
                st.markdown("---")
                
                score = 0
                total = len(st.session_state.current_quiz)
                
                for i, q in enumerate(st.session_state.current_quiz):
                    user_answer = st.session_state.quiz_answers.get(i)
                    correct_answer = q.get('answer')
                    
                    if user_answer == correct_answer:
                        score += 1
                
                percentage = (score / total * 100) if total > 0 else 0
                
                # Display score with styling
                st.markdown(f"""
                <div class='big-score'>
                🎯 Final Score: {score}/{total} ({percentage:.0f}%)
                </div>
                """, unsafe_allow_html=True)
                
                # Celebration for passing
                if percentage >= 60:
                    st.balloons()
                    st.success("🎉 **Congratulations! You passed the quiz!**")
                    st.markdown("Great job! You've demonstrated good understanding of the material.")
                else:
                    st.error("📚 **Keep studying! You can do better!**")
                    st.markdown("Review the material and try again.")
                
                # Show detailed results
                st.markdown("### 📊 Detailed Results:")
                
                for i, q in enumerate(st.session_state.current_quiz):
                    user_answer = st.session_state.quiz_answers.get(i)
                    correct_answer = q.get('answer')
                    is_correct = user_answer == correct_answer
                    
                    with st.expander(f"Question {i+1}: {'✅ Correct' if is_correct else '❌ Incorrect'}"):
                        st.markdown(f"**{q['question']}**")
                        st.markdown(f"Your answer: **{user_answer})** {q['options'].get(user_answer, 'N/A')}")
                        
                        if not is_correct:
                            st.markdown(f"<p class='correct'>Correct answer: {correct_answer}) {q['options'].get(correct_answer, 'N/A')}</p>", 
                                      unsafe_allow_html=True)

if __name__ == "__main__":
    main()