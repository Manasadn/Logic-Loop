import streamlit as st
import PyPDF2
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import string

# -------------- Page Config --------------
st.set_page_config(
    page_title="StudyMate - AI Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- Utility: Stopwords + Keyword Helpers --------------
DEFAULT_STOPWORDS = set("""
a an the and or but if while of to for in on at from by with without into through as is are was were be been being it this that these those than then too very more most much many any some each either neither both all no not only same different do does did done doing can could shall should will would may might must having have has had i me my mine you your yours he him his she her hers they them their theirs we us our ours who whom whose which what when where why how
""".split())

def normalize_text(t: str) -> str:
    return t.lower().translate(str.maketrans('', '', string.punctuation))

def extract_keywords(text: str, min_len: int = 3, max_keywords: int = 12) -> List[str]:
    words = [w for w in normalize_text(text).split() if len(w) >= min_len and w not in DEFAULT_STOPWORDS]
    # simple frequency selection
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # sort by freq then alphabetically
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:max_keywords]]

def count_keyword_hits(text: str, keywords: List[str]) -> Dict[str, int]:
    txt = normalize_text(text)
    counts = {}
    tokens = txt.split()
    for k in keywords:
        counts[k] = tokens.count(k.lower())
    return counts

def highlight_text(text: str, keywords: List[str]) -> str:
    # safe, simple highlight with <mark>
    def repl_token(tok):
        if tok.lower().strip(string.punctuation) in {k.lower() for k in keywords}:
            return f"<mark>{tok}</mark>"
        return tok
    tokens = text.split()
    return " ".join(repl_token(t) for t in tokens)

# -------------- Core System --------------
class StudyMateSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 1000
        self.overlap = 200
        self.question_types = ['multiple_choice', 'true_false', 'short_answer', 'fill_blank']
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks for better context retrieval"""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': current_length
                    })
                    chunk_id += 1
                    
                    # Create overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = (overlap_text + " " + sentence).strip()
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = len(sentence)
            else:
                current_chunk = (current_chunk + " " + sentence).strip()
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': current_length
            })
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict], embeddings: np.ndarray, top_k: int = 5) -> Tuple[List[Dict], np.ndarray]:
        """Find most relevant chunks for the question; returns top_k and all similarities"""
        question_embedding = self.model.encode([question], normalize_embeddings=True)
        similarities = cosine_similarity(question_embedding, embeddings)[0]  # shape: (num_chunks,)
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append({
                'chunk': chunks[idx],
                'similarity': float(similarities[idx]),
                'chunk_id': int(idx)
            })
        return relevant_chunks, similarities
    
    def generate_answer(self, question: str, relevant_chunks: List[Dict], pdf_name: str) -> str:
        """Generate detailed answer using relevant chunks (mock, replace with LLM if desired)"""
        context = "\n\n".join([chunk['chunk']['text'] for chunk in relevant_chunks])
        
        # In production, call your LLM here.
        return self.create_mock_response(question, context, pdf_name)
    
    def create_mock_response(self, question: str, context: str, pdf_name: str) -> str:
        """Create a mock educational response (replace with actual AI API call)"""
        response = f"""Based on the content from "{pdf_name}", here's a detailed explanation:

**Key Concept Analysis:**
Your question: “{question}”

**Detailed Explanation:**
From the text, several passages relate to your question. Let’s break it down clearly and step-by-step.

**Main Points from the Document:**
• The context shows: {context[:240]}...

**Educational Guidance:**
1) Identify the core definitions and principles mentioned.
2) Relate them to your question’s focus.
3) Use the referenced sections (below) for precise phrasing and examples.

**Study Tips:**
- Re-read the highlighted lines in the referenced chunks.
- Summarize each chunk in your own words.
- Try a quick example problem based on those lines.
"""
        return response
    
    # ------------ Question Generation (mock) ------------
    def generate_exam_questions(self, chunks: List[Dict], num_questions: int = 10, difficulty: str = "medium") -> List[Dict]:
        questions = []
        if not chunks:
            return questions
        selected_chunks = random.sample(chunks, min(len(chunks), max(1, num_questions * 2)))
        
        for i in range(min(num_questions, len(selected_chunks))):
            chunk = selected_chunks[i]
            question_type = random.choice(self.question_types)
            
            if question_type == 'multiple_choice':
                question = self.create_multiple_choice_question(chunk, difficulty)
            elif question_type == 'true_false':
                question = self.create_true_false_question(chunk, difficulty)
            elif question_type == 'short_answer':
                question = self.create_short_answer_question(chunk, difficulty)
            else:
                question = self.create_fill_blank_question(chunk, difficulty)
            
            if question:
                questions.append(question)
        return questions[:num_questions]
    
    def create_multiple_choice_question(self, chunk: Dict, difficulty: str) -> Dict:
        content = chunk['text']
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 40]
        if len(sentences) < 1:
            return None
        key_sentence = random.choice(sentences)
        
        if difficulty == "easy":
            stems = [
                "According to the text, what is stated about",
                "Which of the following is true regarding",
                "The document mentions that"
            ]
        elif difficulty == "medium":
            stems = [
                "Based on the information provided, which option best describes",
                "According to the passage, the central idea regarding",
                "Which statement best aligns with"
            ]
        else:
            stems = [
                "From a critical reading of the passage, what can be inferred about",
                "The underlying principle in the text supports which statement about",
                "Considering the passage, which nuanced conclusion holds for"
            ]
        
        question_text = f"{random.choice(stems)} the topic discussed?"
        correct_answer = "It agrees with the key concepts presented in the passage."
        options = [
            correct_answer,
            "It contradicts the main principles.",
            "It is unrelated to the content.",
            "The passage explicitly rejects it."
        ]
        random.shuffle(options)
        return {
            'type': 'multiple_choice',
            'question': question_text,
            'options': options,
            'correct_answer': correct_answer,
            'explanation': f"Supported by: '{key_sentence[:120]}...'",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def create_true_false_question(self, chunk: Dict, difficulty: str) -> Dict:
        content = chunk['text']
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 30]
        if not sentences:
            return None
        base_sentence = random.choice(sentences)
        is_true = random.choice([True, False])
        if is_true:
            question_text = base_sentence
        else:
            question_text = base_sentence + " (This claim contradicts the text.)"
        return {
            'type': 'true_false',
            'question': question_text,
            'correct_answer': is_true,
            'explanation': f"This statement is {'true' if is_true else 'false'} based on the document content.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def create_short_answer_question(self, chunk: Dict, difficulty: str) -> Dict:
        content = chunk['text']
        starters = [
            "Explain the main concept of",
            "Describe how",
            "What is the significance of",
            "Define the term",
            "Summarize the key points about"
        ]
        words = content.split()
        key_phrases = [' '.join(words[i:i+3]) for i in range(0, max(0, len(words)-2), 12)][:3]
        return {
            'type': 'short_answer',
            'question': f"{random.choice(starters)} the topic discussed in this section?",
            'sample_answer': f"Key points include: {', '.join(key_phrases)}...",
            'explanation': "A good answer should cover the main concepts presented in this section.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def create_fill_blank_question(self, chunk: Dict, difficulty: str) -> Dict:
        content = chunk['text']
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 40]
        if not sentences:
            return None
        sentence = random.choice(sentences)
        words = sentence.split()
        important = [w for w in words if len(w.strip(string.punctuation)) > 4 and w.lower() not in DEFAULT_STOPWORDS]
        if not important:
            return None
        word_to_blank = random.choice(important)
        question_sentence = sentence.replace(word_to_blank, "______", 1)
        return {
            'type': 'fill_blank',
            'question': f"Fill in the blank: {question_sentence}",
            'correct_answer': word_to_blank.strip(string.punctuation),
            'explanation': f"The correct word is '{word_to_blank}' based on the context.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def grade_exam(self, questions: List[Dict], user_answers: Dict) -> Dict:
        total_questions = len(questions)
        correct_answers = 0
        feedback = []
        
        for i, question in enumerate(questions):
            user_answer = user_answers.get(f"q_{i}", "")
            if question['type'] == 'multiple_choice':
                is_correct = user_answer == question['correct_answer']
            elif question['type'] == 'true_false':
                is_correct = str(user_answer).lower() == str(question['correct_answer']).lower()
            elif question['type'] == 'fill_blank':
                is_correct = user_answer.lower().strip() == str(question['correct_answer']).lower().strip()
            else:
                is_correct = len(str(user_answer).strip()) > 20
            if is_correct:
                correct_answers += 1
            feedback.append({
                'question_num': i + 1,
                'question': question['question'],
                'user_answer': user_answer,
                'correct_answer': question.get('correct_answer', question.get('sample_answer', '')),
                'is_correct': is_correct,
                'explanation': question['explanation'],
                'type': question['type']
            })
        
        score = (correct_answers / total_questions) * 100 if total_questions else 0.0
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            'score': score,
            'grade': grade,
            'correct': correct_answers,
            'total': total_questions,
            'feedback': feedback
        }

# -------------- Sample PDFs Meta --------------
def load_sample_pdfs():
    return {
        "Physics Fundamentals": {
            "description": "Basic physics concepts including Newton's Laws, energy, and motion",
            "sample_questions": [
                "Explain Newton's Third Law with examples",
                "What is the difference between kinetic and potential energy?",
                "How does friction affect motion?"
            ]
        },
        "Introduction to Biology": {
            "description": "Cell structure, genetics, and basic biological processes",
            "sample_questions": [
                "Describe the structure and function of mitochondria",
                "What is DNA replication?",
                "How does photosynthesis work?"
            ]
        },
        "Mathematics - Calculus": {
            "description": "Limits, derivatives, and integration concepts",
            "sample_questions": [
                "What is a derivative and how is it calculated?",
                "Explain the fundamental theorem of calculus",
                "How do you find the area under a curve?"
            ]
        }
    }

# -------------- Visual Mapping Components --------------
def visualize_similarity_bars(similarities: np.ndarray, top_k_items: List[Dict]):
    # Build dataframe-like lists
    indices = [item['chunk_id'] for item in top_k_items]
    sims = [item['similarity'] for item in top_k_items]
    labels = [f"Chunk {i}" for i in indices]
    fig = px.bar(
        x=sims,
        y=labels,
        orientation='h',
        title="Top Relevant Chunks (Similarity)",
        labels={'x': 'Similarity', 'y': 'Chunk'},
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

def visualize_keyword_sankey(question: str, relevant_chunks: List[Dict]):
    # Extract keywords from the question
    keywords = extract_keywords(question, min_len=3, max_keywords=10)
    if not keywords:
        st.info("No strong keywords found in the question for concept mapping.")
        return
    
    # Build nodes: keywords first, then chunks
    kw_nodes = [f"k:{k}" for k in keywords]
    ch_nodes = [f"c:{rc['chunk_id']}" for rc in relevant_chunks]
    node_labels = keywords + [f"Chunk {rc['chunk_id']}" for rc in relevant_chunks]
    
    # Map node name -> index
    total_nodes = kw_nodes + ch_nodes
    node_index = {name: i for i, name in enumerate(total_nodes)}
    
    # Build links: keyword -> chunk with weight = count
    sources, targets, values, link_labels = [], [], [], []
    for k in keywords:
        for rc in relevant_chunks:
            counts = count_keyword_hits(rc['chunk']['text'], [k])
            val = counts.get(k, 0)
            if val > 0:
                sources.append(node_index[f"k:{k}"])
                targets.append(node_index[f"c:{rc['chunk_id']}"])
                values.append(val)
                link_labels.append(f"{k} → Chunk {rc['chunk_id']}: {val}")
    
    if not values:
        st.info("No keyword occurrences found within the referenced chunks.")
        return
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=16,
            thickness=18,
            line=dict(width=0.5),
            label=node_labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels
        )
    )])
    fig.update_layout(title_text="Concept Map: Question Keywords → Referenced Chunks", height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def show_highlighted_chunks(relevant_chunks: List[Dict], question: str):
    st.markdown("### 📖 Referenced Document Sections (with highlights)")
    keywords = extract_keywords(question, min_len=3, max_keywords=12)
    for j, chunk_info in enumerate(relevant_chunks):
        similarity_pct = chunk_info['similarity'] * 100
        chunk_text = chunk_info['chunk']['text']
        snippet = chunk_text if len(chunk_text) < 1200 else chunk_text[:1200] + "..."
        highlighted = highlight_text(snippet, keywords)
        st.markdown(f"**Section {j+1} — Chunk {chunk_info['chunk_id']} (Relevance: {similarity_pct:.1f}%)**")
        st.markdown(
            f"<div style='background:#0f172a0a;border-radius:12px;padding:12px'>{highlighted}</div>",
            unsafe_allow_html=True
        )
        st.markdown("")

# -------------- Streamlit UI Layers --------------
def main():
    # Initialize session state
    if 'study_system' not in st.session_state:
        st.session_state.study_system = StudyMateSystem()
    if 'current_pdf_data' not in st.session_state:
        st.session_state.current_pdf_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_exam' not in st.session_state:
        st.session_state.current_exam = None
    if 'exam_answers' not in st.session_state:
        st.session_state.exam_answers = {}
    if 'exam_results' not in st.session_state:
        st.session_state.exam_results = None
    if 'exam_history' not in st.session_state:
        st.session_state.exam_history = []

    st.title("📚 StudyMate - Your AI-Powered Study Companion")
    
    tab1, tab2 = st.tabs(["💬 Q&A Mode", "📝 Exam Mode"])
    with tab1:
        qa_mode()
    with tab2:
        exam_mode()

def qa_mode():
    """Q&A Mode Interface"""
    st.markdown("Upload any PDF document and ask detailed questions. I'll help you understand the material like a personal tutor!")

    with st.sidebar:
        st.header("📄 Document Management")
        uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing your PDF..."):
                    text = st.session_state.study_system.extract_text_from_pdf(uploaded_file)
                    if text:
                        chunks = st.session_state.study_system.chunk_text(text)
                        embeddings = st.session_state.study_system.create_embeddings(chunks)
                        st.session_state.current_pdf_data = {
                            'name': uploaded_file.name,
                            'text': text,
                            'chunks': chunks,
                            'embeddings': embeddings,
                            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.success(f"✅ PDF processed successfully!")
                        st.info(f"Document: {uploaded_file.name}")
                        st.info(f"Text chunks created: {len(chunks)}")
        
        st.header("📖 Sample Documents")
        st.markdown("Try these example documents:")
        sample_pdfs = load_sample_pdfs()
        for pdf_name, info in sample_pdfs.items():
            if st.button(f"📄 {pdf_name}", key=f"sample_{pdf_name}"):
                st.session_state.current_pdf_data = {
                    'name': f"{pdf_name}.pdf",
                    'text': f"Sample content for {pdf_name}. This is a demonstration of the StudyMate system with placeholder text covering key ideas, definitions, and examples to showcase the visual mapping.",
                    'chunks': [{'id': 0, 'text': f"Sample content for {pdf_name}. This section discusses key definitions and relationships, emphasizing concepts and examples for learning.", 'length': 160}],
                    'embeddings': np.random.rand(1, 384),
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'is_sample': True
                }
                st.success(f"Sample PDF '{pdf_name}' loaded!")
        
        if st.session_state.current_pdf_data:
            st.header("📋 Current Document")
            st.info(f"**Document:** {st.session_state.current_pdf_data['name']}")
            st.info(f"**Processed:** {st.session_state.current_pdf_data['processed_at']}")
            if st.button("Clear Document"):
                st.session_state.current_pdf_data = None
                st.session_state.chat_history = []
                st.session_state.current_exam = None
                st.session_state.exam_answers = {}
                st.session_state.exam_results = None
                st.rerun()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("💬 Ask Your Questions")
        if st.session_state.current_pdf_data:
            question = st.text_area(
                "What would you like to know about this document?",
                height=100,
                placeholder="Example: Explain Newton's Third Law with examples"
            )
            if st.button("Get Answer", type="primary", disabled=not question):
                with st.spinner("Thinking about your question..."):
                    relevant_chunks, similarities = st.session_state.study_system.find_relevant_chunks(
                        question,
                        st.session_state.current_pdf_data['chunks'],
                        st.session_state.current_pdf_data['embeddings'],
                        top_k=5
                    )
                    answer = st.session_state.study_system.generate_answer(
                        question,
                        relevant_chunks,
                        st.session_state.current_pdf_data['name']
                    )
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer,
                        'timestamp': datetime.now().strftime("%H:%M"),
                        'relevant_chunks': relevant_chunks,
                        'similarities': similarities.tolist() if isinstance(similarities, np.ndarray) else similarities
                    })
                    st.success("Answer generated!")

            # Display chat history (newest first)
            if st.session_state.chat_history:
                st.header("📝 Study Session History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    idx = len(st.session_state.chat_history) - i
                    with st.expander(f"Q{idx}: {chat['question'][:60]}... ({chat['timestamp']})", expanded=(i==0)):
                        st.markdown("**Answer:**")
                        st.markdown(chat['answer'])

                        # --- Visual Mapping Section ---
                        st.markdown("## 🧭 Visual Mapping")

                        # 1) Similarity bars
                        st.markdown("#### Top Relevant Chunks")
                        visualize_similarity_bars(
                            similarities=np.array(chat['similarities']),
                            top_k_items=chat['relevant_chunks']
                        )

                        # 2) Sankey concept map
                        st.markdown("#### Concept Map (Question Keywords → Chunks)")
                        visualize_keyword_sankey(
                            question=chat['question'],
                            relevant_chunks=chat['relevant_chunks']
                        )

                        # 3) Highlighted referenced sections
                        show_highlighted_chunks(
                            relevant_chunks=chat['relevant_chunks'],
                            question=chat['question']
                        )
        else:
            st.info("👆 Please upload a PDF document or select a sample document from the sidebar to get started!")
            st.markdown("""
            ### How to use StudyMate:
            1. **Upload a PDF** - Use the sidebar to upload your study material  
            2. **Ask Questions** - Type any question about the document content  
            3. **Visual Mapping** - See which sections your question maps to  
            4. **Deepen Understanding** - Use the highlights and concept map to study
            """)

    with col2:
        st.header("🎯 Quick Help")
        if st.session_state.get('current_pdf_data') and 'is_sample' in st.session_state.current_pdf_data:
            pdf_name_clean = st.session_state.current_pdf_data['name'].replace('.pdf', '')
            sample_pdfs = load_sample_pdfs()
            for name, info in sample_pdfs.items():
                if name in pdf_name_clean:
                    st.markdown("**💡 Try these sample questions:**")
                    for q in info['sample_questions']:
                        if st.button(f"❓ {q}", key=f"q_{q}"):
                            st.session_state.sample_question = q
                            st.experimental_rerun()

        st.markdown("""
        ### 📖 Study Tips:
        - Ask follow-up questions for clarity  
        - Request examples for complex concepts  
        - Use the concept map to trace ideas  
        - Review highlighted snippets to anchor memory
        """)

        if st.session_state.get('current_pdf_data'):
            st.markdown("### 📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))
            st.metric("Document Sections", len(st.session_state.current_pdf_data['chunks']))

def exam_mode():
    """Exam Mode Interface"""
    st.markdown("Generate and take practice exams based on your uploaded PDF content!")
    if not st.session_state.get('current_pdf_data'):
        st.warning("Please upload a PDF document first to generate exam questions.")
        return
    
    st.header("📝 Create Your Exam")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_questions = st.selectbox("Number of Questions", [5, 10, 15, 20], index=1)
    with col2:
        difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"], index=1)
    with col3:
        time_limit = st.selectbox("Time Limit (minutes)", [10, 15, 30, 45, 60], index=2)
    
    if st.button("🎯 Generate New Exam", type="primary"):
        with st.spinner("Generating exam questions..."):
            questions = st.session_state.study_system.generate_exam_questions(
                st.session_state.current_pdf_data['chunks'],
                num_questions=num_questions,
                difficulty=difficulty
            )
            st.session_state.current_exam = {
                'questions': questions,
                'settings': {
                    'num_questions': num_questions,
                    'difficulty': difficulty,
                    'time_limit': time_limit,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            st.session_state.exam_answers = {}
            st.session_state.exam_results = None
        st.success(f"✅ Generated {len(questions)} questions!")
        st.rerun()
    
    if st.session_state.current_exam and not st.session_state.exam_results:
        display_exam()
    elif st.session_state.exam_results:
        display_exam_results()
    
    if st.session_state.get('exam_history'):
        st.header("📊 Exam History")
        for i, exam_record in enumerate(reversed(st.session_state.exam_history[-5:])):
            with st.expander(f"Exam {len(st.session_state.exam_history)-i} - Score: {exam_record['score']:.1f}% ({exam_record['grade']})"):
                st.write(f"**Date:** {exam_record['date']}")
                st.write(f"**Questions:** {exam_record['total_questions']}")
                st.write(f"**Difficulty:** {exam_record['difficulty']}")
                st.write(f"**Correct Answers:** {exam_record['correct']}/{exam_record['total']}")

def display_exam():
    exam = st.session_state.current_exam
    questions = exam['questions']
    st.header(f"📝 Exam ({len(questions)} Questions)")
    st.info(f"Difficulty: {exam['settings']['difficulty'].title()} | Time Limit: {exam['settings']['time_limit']} minutes")
    
    with st.form("exam_form"):
        st.markdown("---")
        for i, q in enumerate(questions):
            st.subheader(f"Question {i+1}")
            st.write(q['question'])
            if q['type'] == 'multiple_choice':
                answer = st.radio(
                    f"Select your answer for Question {i+1}:",
                    q['options'],
                    key=f"q_{i}",
                    index=None
                )
            elif q['type'] == 'true_false':
                answer = st.radio(
                    f"Select your answer for Question {i+1}:",
                    [True, False],
                    key=f"q_{i}",
                    index=None,
                    format_func=lambda x: "True" if x else "False"
                )
            elif q['type'] == 'fill_blank':
                answer = st.text_input(
                    f"Fill in the blank for Question {i+1}:",
                    key=f"q_{i}",
                    placeholder="Enter your answer here..."
                )
            else:
                answer = st.text_area(
                    f"Provide your answer for Question {i+1}:",
                    key=f"q_{i}",
                    height=100,
                    placeholder="Write your detailed answer here..."
                )
            st.markdown("---")
        
        submitted = st.form_submit_button("📋 Submit Exam", type="primary")
        if submitted:
            answers = {f"q_{i}": st.session_state.get(f"q_{i}", "") for i in range(len(questions))}
            results = st.session_state.study_system.grade_exam(questions, answers)
            st.session_state.exam_results = results
            st.session_state.exam_answers = answers
            exam_record = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'score': results['score'],
                'grade': results['grade'],
                'correct': results['correct'],
                'total': results['total'],
                'difficulty': exam['settings']['difficulty'],
                'total_questions': len(questions)
            }
            st.session_state.exam_history.append(exam_record)
            st.rerun()

def display_exam_results():
    results = st.session_state.exam_results
    st.header("🎉 Exam Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score", f"{results['score']:.1f}%")
    with col2:
        st.metric("Grade", results['grade'])
    with col3:
        st.metric("Correct", f"{results['correct']}/{results['total']}")
    with col4:
        accuracy = (results['correct'] / results['total']) * 100 if results['total'] else 0
        st.metric("Accuracy", f"{accuracy:.0f}%")
    
    if results['score'] >= 80:
        st.success("🌟 Excellent work! You have a strong understanding of the material.")
    elif results['score'] >= 70:
        st.info("👍 Good job! You understand most concepts well.")
    elif results['score'] >= 60:
        st.warning("📖 Fair performance. Consider reviewing the material more thoroughly.")
    else:
        st.error("📚 More study needed. Focus on understanding the key concepts.")
    
    st.header("📋 Detailed Feedback")
    for fb in results['feedback']:
        with st.expander(f"Question {fb['question_num']} - {'✅ Correct' if fb['is_correct'] else '❌ Incorrect'}"):
            st.write(f"**Question:** {fb['question']}")
            st.write(f"**Your Answer:** {fb['user_answer']}")
            st.write(f"**Correct Answer:** {fb['correct_answer']}")
            st.write(f"**Explanation:** {fb['explanation']}")
            if fb['is_correct']:
                st.success("Well done! Your answer is correct.")
            else:
                st.error("This answer needs improvement. Review the explanation above.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take New Exam"):
            st.session_state.current_exam = None
            st.session_state.exam_answers = {}
            st.session_state.exam_results = None
            st.rerun()
    with col2:
        if st.button("📚 Back to Study Mode"):
            st.session_state.current_exam = None
            st.session_state.exam_answers = {}
            st.session_state.exam_results = None
            st.rerun()

if __name__ == "__main__":
    main()
