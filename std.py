import streamlit as st
import PyPDF2
import io
import re
from typing import List, Dict, Tuple
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import random
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="StudyMate - AI Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks for better context retrieval"""
        sentences = re.split(r'[.!?]+', text)
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
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = len(sentence)
            else:
                current_chunk += " " + sentence
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
        embeddings = self.model.encode(texts)
        return embeddings
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict], embeddings: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Find most relevant chunks for the question"""
        question_embedding = self.model.encode([question])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_chunks = []
        
        for idx in top_indices:
            relevant_chunks.append({
                'chunk': chunks[idx],
                'similarity': similarities[idx],
                'chunk_id': idx
            })
        
        return relevant_chunks
    
    def generate_answer(self, question: str, relevant_chunks: List[Dict], pdf_name: str) -> str:
        """Generate detailed answer using relevant chunks"""
        context = "\n\n".join([chunk['chunk']['text'] for chunk in relevant_chunks])
        
        # Create a comprehensive prompt for educational Q&A
        prompt = f"""You are StudyMate, an AI tutor helping students understand academic material. 
Based on the following context from the document "{pdf_name}", provide a detailed, educational answer to the student's question.

Context from document:
{context}

Student's Question: {question}

Instructions for your response:
1. Provide a clear, detailed explanation that a student can easily understand
2. Use examples when helpful to illustrate concepts
3. Reference specific parts of the document when relevant
4. Break down complex concepts into simpler parts
5. Be encouraging and supportive like a real tutor
6. If the context doesn't fully answer the question, explain what you can based on the available information

Answer:"""

        try:
            # Note: In a real implementation, you would use OpenAI API here
            # For this demo, we'll create a mock response based on the context
            return self.create_mock_response(question, context, pdf_name)
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
    
    def create_mock_response(self, question: str, context: str, pdf_name: str) -> str:
        """Create a mock educational response (replace with actual AI API call)"""
        response = f"""Based on the content from "{pdf_name}", here's a detailed explanation:

**Key Concept Analysis:**
The document provides relevant information about your question: "{question}"

**Detailed Explanation:**
From the text, I can see that the material covers several important points related to your question. Let me break this down for you:

**Main Points from the Document:**
• The context shows: {context[:200]}...

**Educational Guidance:**
To better understand this concept, consider these key aspects:
1. The fundamental principles outlined in the document
2. How these concepts connect to broader topics in the subject
3. Practical applications you might encounter

**Study Tips:**
- Review the specific sections I referenced above
- Try to connect these concepts to examples you know
- Practice applying these principles to similar problems

Would you like me to elaborate on any specific aspect of this explanation?"""
        
        return response
    
    def generate_exam_questions(self, chunks: List[Dict], num_questions: int = 10, difficulty: str = "medium") -> List[Dict]:
        """Generate various types of exam questions from PDF content"""
        questions = []
        selected_chunks = random.sample(chunks, min(len(chunks), num_questions * 2))
        
        for i in range(min(num_questions, len(selected_chunks))):
            chunk = selected_chunks[i]
            question_type = random.choice(self.question_types)
            
            if question_type == 'multiple_choice':
                question = self.create_multiple_choice_question(chunk, difficulty)
            elif question_type == 'true_false':
                question = self.create_true_false_question(chunk, difficulty)
            elif question_type == 'short_answer':
                question = self.create_short_answer_question(chunk, difficulty)
            else:  # fill_blank
                question = self.create_fill_blank_question(chunk, difficulty)
            
            if question:
                questions.append(question)
        
        return questions[:num_questions]
    
    def create_multiple_choice_question(self, chunk: Dict, difficulty: str) -> Dict:
        """Create a multiple choice question from a text chunk"""
        content = chunk['text']
        
        # Extract key concepts (simplified approach)
        sentences = content.split('.')
        if len(sentences) < 2:
            return None
            
        key_sentence = random.choice([s.strip() for s in sentences if len(s.strip()) > 50])
        
        # Generate question based on difficulty
        if difficulty == "easy":
            question_stems = [
                f"According to the text, what is mentioned about",
                f"The document states that",
                f"Which of the following is true based on the content"
            ]
        elif difficulty == "medium":
            question_stems = [
                f"Based on the information provided, which statement best describes",
                f"The text suggests that",
                f"According to the passage, the main concept relates to"
            ]
        else:  # hard
            question_stems = [
                f"Analyzing the given information, what can be inferred about",
                f"The underlying principle described in the text indicates that",
                f"Based on critical analysis of the content"
            ]
        
        # Mock question generation (replace with actual AI)
        question_text = f"{random.choice(question_stems)} the topic discussed?"
        
        # Generate options (simplified)
        correct_answer = "The information aligns with the key concepts presented"
        options = [
            correct_answer,
            "This contradicts the main principles outlined",
            "The text does not provide sufficient information",
            "This represents an alternative interpretation"
        ]
        random.shuffle(options)
        
        return {
            'type': 'multiple_choice',
            'question': question_text,
            'options': options,
            'correct_answer': correct_answer,
            'explanation': f"Based on the text: '{key_sentence[:100]}...', this answer is supported by the content.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def create_true_false_question(self, chunk: Dict, difficulty: str) -> Dict:
        """Create a true/false question from a text chunk"""
        content = chunk['text']
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]
        
        if not sentences:
            return None
            
        base_sentence = random.choice(sentences)
        
        # Create true or false statement
        is_true = random.choice([True, False])
        
        if is_true:
            question_text = base_sentence
        else:
            # Create a false statement by modifying the original
            false_modifications = [
                "The opposite of what is stated is true",
                "This contradicts the main principles",
                "This is not supported by the evidence"
            ]
            question_text = f"{base_sentence} ({random.choice(false_modifications)})"
        
        return {
            'type': 'true_false',
            'question': question_text,
            'correct_answer': is_true,
            'explanation': f"This statement is {'true' if is_true else 'false'} based on the document content.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def create_short_answer_question(self, chunk: Dict, difficulty: str) -> Dict:
        """Create a short answer question from a text chunk"""
        content = chunk['text']
        
        question_starters = [
            "Explain the main concept of",
            "Describe how",
            "What is the significance of",
            "Define the term",
            "Summarize the key points about"
        ]
        
        question_text = f"{random.choice(question_starters)} the topic discussed in this section?"
        
        # Extract key phrases for sample answer
        words = content.split()
        key_phrases = [' '.join(words[i:i+3]) for i in range(0, len(words)-2, 10)][:3]
        
        return {
            'type': 'short_answer',
            'question': question_text,
            'sample_answer': f"Key points include: {', '.join(key_phrases)}...",
            'explanation': "A good answer should cover the main concepts presented in this section.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def create_fill_blank_question(self, chunk: Dict, difficulty: str) -> Dict:
        """Create a fill-in-the-blank question from a text chunk"""
        content = chunk['text']
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 40]
        
        if not sentences:
            return None
            
        sentence = random.choice(sentences)
        words = sentence.split()
        
        if len(words) < 5:
            return None
        
        # Remove a key word (not articles, prepositions, etc.)
        important_words = [w for w in words if len(w) > 4 and w.lower() not in ['that', 'with', 'from', 'they', 'have', 'this', 'will', 'been']]
        
        if not important_words:
            return None
            
        word_to_blank = random.choice(important_words)
        question_sentence = sentence.replace(word_to_blank, "______", 1)
        
        return {
            'type': 'fill_blank',
            'question': f"Fill in the blank: {question_sentence}",
            'correct_answer': word_to_blank,
            'explanation': f"The correct word is '{word_to_blank}' based on the context of the sentence.",
            'difficulty': difficulty,
            'chunk_id': chunk['id']
        }
    
    def grade_exam(self, questions: List[Dict], user_answers: Dict) -> Dict:
        """Grade the exam and provide detailed feedback"""
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
                is_correct = user_answer.lower().strip() == question['correct_answer'].lower().strip()
            else:  # short_answer
                # For short answers, we'll do a simple keyword check (in real app, use AI)
                is_correct = len(user_answer.strip()) > 20  # Basic length check
            
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
        
        score = (correct_answers / total_questions) * 100
        
        # Determine grade
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

def load_sample_pdfs():
    """Load information about sample PDFs (in a real app, these would be actual files)"""
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

    # Header with navigation
    st.title("📚 StudyMate - Your AI-Powered Study Companion")
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["💬 Q&A Mode", "📝 Exam Mode"])
    
    with tab1:
        qa_mode()
    
    with tab2:
        exam_mode()

def qa_mode():
    """Q&A Mode Interface"""
    st.markdown("Upload any PDF document and ask detailed questions. I'll help you understand the material like a personal tutor!")

    # Sidebar for PDF upload and management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload your PDF document",
            type=['pdf'],
            help="Upload any academic PDF and start asking questions!"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing your PDF..."):
                    # Extract text
                    text = st.session_state.study_system.extract_text_from_pdf(uploaded_file)
                    if text:
                        # Create chunks and embeddings
                        chunks = st.session_state.study_system.chunk_text(text)
                        embeddings = st.session_state.study_system.create_embeddings(chunks)
                        
                        # Store in session state
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
        
        # Sample PDFs section
        st.header("📖 Sample Documents")
        st.markdown("Try these example documents:")
        
        sample_pdfs = load_sample_pdfs()
        for pdf_name, info in sample_pdfs.items():
            if st.button(f"📄 {pdf_name}", key=f"sample_{pdf_name}"):
                # Simulate loading a sample PDF
                st.session_state.current_pdf_data = {
                    'name': f"{pdf_name}.pdf",
                    'text': f"Sample content for {pdf_name}. This is a demonstration of the StudyMate system.",
                    'chunks': [{'id': 0, 'text': f"Sample content for {pdf_name}", 'length': 100}],
                    'embeddings': np.random.rand(1, 384),  # Mock embedding
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'is_sample': True
                }
                st.success(f"Sample PDF '{pdf_name}' loaded!")
        
        # Current document info
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

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Questions")
        
        if st.session_state.current_pdf_data:
            # Question input
            question = st.text_area(
                "What would you like to know about this document?",
                height=100,
                placeholder="Example: Explain Newton's Third Law with examples"
            )
            
            if st.button("Get Answer", type="primary", disabled=not question):
                if question:
                    with st.spinner("Thinking about your question..."):
                        # Find relevant chunks
                        relevant_chunks = st.session_state.study_system.find_relevant_chunks(
                            question,
                            st.session_state.current_pdf_data['chunks'],
                            st.session_state.current_pdf_data['embeddings']
                        )
                        
                        # Generate answer
                        answer = st.session_state.study_system.generate_answer(
                            question,
                            relevant_chunks,
                            st.session_state.current_pdf_data['name']
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'timestamp': datetime.now().strftime("%H:%M"),
                            'relevant_chunks': relevant_chunks
                        })
            
            # Display chat history
            if st.session_state.chat_history:
                st.header("📝 Study Session History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}... ({chat['timestamp']})"):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown("**Answer:**")
                        st.markdown(chat['answer'])
                        
                        # Show relevant document sections
                        if chat['relevant_chunks']:
                            st.markdown("**📖 Referenced Document Sections:**")
                            for j, chunk_info in enumerate(chat['relevant_chunks'][:2]):  # Show top 2
                                similarity_pct = chunk_info['similarity'] * 100
                                st.markdown(f"*Section {j+1} (Relevance: {similarity_pct:.1f}%):*")
                                st.markdown(f"> {chunk_info['chunk']['text'][:200]}...")
        else:
            st.info("👆 Please upload a PDF document or select a sample document from the sidebar to get started!")
            
            # Show welcome message with instructions
            st.markdown("""
            ### How to use StudyMate:
            
            1. **Upload a PDF** - Use the sidebar to upload your study material
            2. **Ask Questions** - Type any question about the document content  
            3. **Get Detailed Answers** - Receive explanations with document references
            4. **Build Understanding** - Use follow-up questions to deepen your knowledge
            
            ### Example Questions:
            - "Explain the main concept in Chapter 3"
            - "What are the key formulas mentioned?"
            - "Can you give me examples of this theory?"
            - "How does this relate to what I learned earlier?"
            """)
    
    with col2:
        st.header("🎯 Quick Help")
        
        if st.session_state.current_pdf_data and 'is_sample' in st.session_state.current_pdf_data:
            # Show sample questions for the current sample PDF
            pdf_name_clean = st.session_state.current_pdf_data['name'].replace('.pdf', '')
            sample_pdfs = load_sample_pdfs()
            
            for name, info in sample_pdfs.items():
                if name in pdf_name_clean:
                    st.markdown("**💡 Try these sample questions:**")
                    for question in info['sample_questions']:
                        if st.button(f"❓ {question}", key=f"q_{question}"):
                            st.session_state.sample_question = question
        
        # Study tips
        st.markdown("""
        ### 📖 Study Tips:
        - Ask follow-up questions for clarity
        - Request examples for complex concepts  
        - Break down difficult topics into parts
        - Ask for connections between concepts
        
        ### 🔍 Question Types:
        - **Explain:** Get detailed explanations
        - **Compare:** Understand differences
        - **Examples:** See practical applications
        - **Summarize:** Get key points
        """)
        
        # Usage statistics
        if st.session_state.current_pdf_data:
            st.markdown("### 📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))
            st.metric("Document Sections", len(st.session_state.current_pdf_data['chunks']))

def exam_mode():
    """Exam Mode Interface"""
    st.markdown("Generate and take practice exams based on your uploaded PDF content!")
    
    if not st.session_state.current_pdf_data:
        st.warning("Please upload a PDF document first to generate exam questions.")
        return
    
    # Exam configuration
    st.header("📝 Create Your Exam")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_questions = st.selectbox("Number of Questions", [5, 10, 15, 20], index=1)
    with col2:
        difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"], index=1)
    with col3:
        time_limit = st.selectbox("Time Limit (minutes)", [10, 15, 30, 45, 60], index=2)
    
    # Generate exam button
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
    
    # Display current exam
    if st.session_state.current_exam and not st.session_state.exam_results:
        display_exam()
    
    # Display exam results
    elif st.session_state.exam_results:
        display_exam_results()
    
    # Exam history
    if st.session_state.exam_history:
        st.header("📊 Exam History")
        for i, exam_record in enumerate(reversed(st.session_state.exam_history[-5:])):  # Show last 5
            with st.expander(f"Exam {len(st.session_state.exam_history)-i} - Score: {exam_record['score']:.1f}% ({exam_record['grade']})"):
                st.write(f"**Date:** {exam_record['date']}")
                st.write(f"**Questions:** {exam_record['total_questions']}")
                st.write(f"**Difficulty:** {exam_record['difficulty']}")
                st.write(f"**Correct Answers:** {exam_record['correct']}/{exam_record['total']}")

def display_exam():
    """Display the current exam questions"""
    exam = st.session_state.current_exam
    questions = exam['questions']
    
    st.header(f"📝 Exam ({len(questions)} Questions)")
    st.info(f"Difficulty: {exam['settings']['difficulty'].title()} | Time Limit: {exam['settings']['time_limit']} minutes")
    
    with st.form("exam_form"):
        st.markdown("---")
        
        for i, question in enumerate(questions):
            st.subheader(f"Question {i+1}")
            st.write(question['question'])
            
            if question['type'] == 'multiple_choice':
                answer = st.radio(
                    f"Select your answer for Question {i+1}:",
                    question['options'],
                    key=f"q_{i}",
                    index=None
                )
                
            elif question['type'] == 'true_false':
                answer = st.radio(
                    f"Select your answer for Question {i+1}:",
                    [True, False],
                    key=f"q_{i}",
                    index=None,
                    format_func=lambda x: "True" if x else "False"
                )
                
            elif question['type'] == 'fill_blank':
                answer = st.text_input(
                    f"Fill in the blank for Question {i+1}:",
                    key=f"q_{i}",
                    placeholder="Enter your answer here..."
                )
                
            else:  # short_answer
                answer = st.text_area(
                    f"Provide your answer for Question {i+1}:",
                    key=f"q_{i}",
                    height=100,
                    placeholder="Write your detailed answer here..."
                )
            
            st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button("📋 Submit Exam", type="primary")
        
        if submitted:
            # Collect answers
            answers = {}
            for i in range(len(questions)):
                answers[f"q_{i}"] = st.session_state.get(f"q_{i}", "")
            
            # Grade the exam
            results = st.session_state.study_system.grade_exam(questions, answers)
            
            # Store results
            st.session_state.exam_results = results
            st.session_state.exam_answers = answers
            
            # Add to history
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
    """Display exam results and feedback"""
    results = st.session_state.exam_results
    
    st.header("🎉 Exam Results")
    
    # Score display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score", f"{results['score']:.1f}%")
    with col2:
        st.metric("Grade", results['grade'])
    with col3:
        st.metric("Correct", f"{results['correct']}/{results['total']}")
    with col4:
        accuracy = (results['correct'] / results['total']) * 100
        st.metric("Accuracy", f"{accuracy:.0f}%")
    
    # Performance indicator
    if results['score'] >= 80:
        st.success("🌟 Excellent work! You have a strong understanding of the material.")
    elif results['score'] >= 70:
        st.info("👍 Good job! You understand most concepts well.")
    elif results['score'] >= 60:
        st.warning("📖 Fair performance. Consider reviewing the material more thoroughly.")
    else:
        st.error("📚 More study needed. Focus on understanding the key concepts.")
    
    # Detailed feedback
    st.header("📋 Detailed Feedback")
    
    for feedback in results['feedback']:
        with st.expander(f"Question {feedback['question_num']} - {'✅ Correct' if feedback['is_correct'] else '❌ Incorrect'}"):
            st.write(f"**Question:** {feedback['question']}")
            st.write(f"**Your Answer:** {feedback['user_answer']}")
            st.write(f"**Correct Answer:** {feedback['correct_answer']}")
            st.write(f"**Explanation:** {feedback['explanation']}")
            
            if feedback['is_correct']:
                st.success("Well done! Your answer is correct.")
            else:
                st.error("This answer needs improvement. Review the explanation above.")
    
    # Action buttons
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