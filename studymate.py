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

    # Header
    st.title("📚 StudyMate - Your AI-Powered Study Companion")
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

if __name__ == "__main__":
    main()