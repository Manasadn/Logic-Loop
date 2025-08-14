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
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import networkx as nx
from PIL import Image
import base64

# Configure page
st.set_page_config(
    page_title="StudyMate - AI Learning Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}

.progress-ring {
    transform: rotate(-90deg);
}

.confidence-high { color: #28a745; }
.confidence-medium { color: #ffc107; }
.confidence-low { color: #dc3545; }

.tutor-personality {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border: 2px solid #e9ecef;
}

.tutor-friendly { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); }
.tutor-academic { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
.tutor-enthusiastic { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); }
.tutor-patient { background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); color: white; }
</style>
""", unsafe_allow_html=True)

class AdvancedStudyMateSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 1000
        self.overlap = 200
        self.question_types = ['multiple_choice', 'true_false', 'short_answer', 'fill_blank']
        
        # Student progress tracking
        self.learning_levels = ['beginner', 'intermediate', 'advanced']
        self.personality_types = {
            'friendly': {
                'name': '😊 Friendly Tutor',
                'style': 'Warm, encouraging, uses simple language',
                'greeting': "Hi there! I'm excited to help you learn today! 🌟"
            },
            'academic': {
                'name': '🎓 Academic Expert', 
                'style': 'Professional, detailed, scholarly approach',
                'greeting': "Good day. I shall provide comprehensive academic guidance."
            },
            'enthusiastic': {
                'name': '🚀 Enthusiastic Coach',
                'style': 'Energetic, motivational, uses examples',
                'greeting': "Let's dive into learning and make it awesome! 🎉"
            },
            'patient': {
                'name': '🧘 Patient Guide',
                'style': 'Calm, methodical, breaks down complex topics',
                'greeting': "Take your time. We'll work through this step by step."
            }
        }
        
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
        """Split text into overlapping chunks with enhanced metadata"""
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
                    # Analyze chunk complexity
                    complexity = self.analyze_complexity(current_chunk)
                    key_concepts = self.extract_key_concepts(current_chunk)
                    
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': current_length,
                        'complexity': complexity,
                        'key_concepts': key_concepts,
                        'topics': self.identify_topics(current_chunk)
                    })
                    chunk_id += 1
                    
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
            complexity = self.analyze_complexity(current_chunk)
            key_concepts = self.extract_key_concepts(current_chunk)
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': current_length,
                'complexity': complexity,
                'key_concepts': key_concepts,
                'topics': self.identify_topics(current_chunk)
            })
        
        return chunks
    
    def analyze_complexity(self, text: str) -> str:
        """Analyze text complexity level"""
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words])
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Simple complexity scoring
        complexity_score = (avg_word_length * 0.5) + (avg_sentence_length * 0.3)
        
        if complexity_score > 15:
            return 'advanced'
        elif complexity_score > 10:
            return 'intermediate'
        else:
            return 'beginner'
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction (replace with more sophisticated NLP)
        words = re.findall(r'\b[A-Z][a-z]*\b', text)  # Capitalized words
        technical_terms = re.findall(r'\b\w{6,}\b', text.lower())  # Longer technical terms
        return list(set(words + technical_terms[:5]))[:10]  # Top 10 concepts
    
    def identify_topics(self, text: str) -> List[str]:
        """Identify main topics in the text"""
        # Topic keywords mapping
        topic_keywords = {
            'physics': ['force', 'energy', 'motion', 'velocity', 'acceleration', 'newton', 'gravity'],
            'biology': ['cell', 'organism', 'dna', 'evolution', 'photosynthesis', 'enzyme'],
            'chemistry': ['molecule', 'atom', 'reaction', 'compound', 'element', 'bond'],
            'mathematics': ['equation', 'function', 'derivative', 'integral', 'theorem', 'proof'],
            'history': ['century', 'war', 'empire', 'revolution', 'ancient', 'medieval'],
            'literature': ['character', 'theme', 'metaphor', 'author', 'novel', 'poetry']
        }
        
        text_lower = text.lower()
        identified_topics = []
        
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches >= 2:  # Topic threshold
                identified_topics.append(topic)
        
        return identified_topics[:3]  # Top 3 topics
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict], embeddings: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Find most relevant chunks with confidence scoring"""
        question_embedding = self.model.encode([question])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_chunks = []
        
        for idx in top_indices:
            confidence = self.calculate_confidence(similarities[idx], chunks[idx])
            relevant_chunks.append({
                'chunk': chunks[idx],
                'similarity': similarities[idx],
                'chunk_id': idx,
                'confidence': confidence
            })
        
        return relevant_chunks
    
    def calculate_confidence(self, similarity: float, chunk: Dict) -> Dict:
        """Calculate AI confidence in the answer"""
        # Multi-factor confidence calculation
        similarity_factor = similarity * 0.4
        complexity_match = 0.2 if chunk.get('complexity') else 0.1
        content_length = min(len(chunk['text']) / 1000, 0.3)  # Normalize to max 0.3
        concept_richness = min(len(chunk.get('key_concepts', [])) / 10, 0.1)
        
        total_confidence = similarity_factor + complexity_match + content_length + concept_richness
        
        if total_confidence > 0.7:
            level = 'high'
        elif total_confidence > 0.5:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'score': total_confidence,
            'level': level,
            'factors': {
                'similarity': similarity_factor,
                'complexity': complexity_match,
                'content_length': content_length,
                'concept_richness': concept_richness
            }
        }
    
    def generate_adaptive_answer(self, question: str, relevant_chunks: List[Dict], 
                               student_level: str, personality: str, pdf_name: str) -> Dict:
        """Generate personalized answer based on student level and AI personality"""
        context = "\n\n".join([chunk['chunk']['text'] for chunk in relevant_chunks])
        avg_confidence = np.mean([chunk['confidence']['score'] for chunk in relevant_chunks])
        
        # Adapt explanation based on student level
        if student_level == 'beginner':
            complexity_instruction = "Use simple language, provide basic explanations, include analogies"
        elif student_level == 'intermediate':
            complexity_instruction = "Use moderate complexity, include some technical terms with explanations"
        else:
            complexity_instruction = "Use advanced terminology, provide detailed analysis, assume prior knowledge"
        
        # Apply personality style
        personality_style = self.personality_types[personality]['style']
        
        # Generate concept map data
        concept_map = self.create_concept_map(relevant_chunks)
        
        # Create mock response (replace with actual AI API)
        answer = self.create_adaptive_response(question, context, student_level, personality, pdf_name)
        
        # Generate follow-up questions for active recall
        follow_up_questions = self.generate_follow_up_questions(context, student_level)
        
        # Suggest related concepts
        related_concepts = self.suggest_related_concepts(relevant_chunks)
        
        return {
            'answer': answer,
            'confidence': avg_confidence,
            'concept_map': concept_map,
            'follow_up_questions': follow_up_questions,
            'related_concepts': related_concepts,
            'source_chunks': relevant_chunks,
            'learning_level': student_level,
            'personality_used': personality
        }
    
    def create_concept_map(self, relevant_chunks: List[Dict]) -> Dict:
        """Create concept map data for visualization"""
        nodes = []
        edges = []
        
        # Create nodes from key concepts
        all_concepts = []
        for chunk in relevant_chunks:
            concepts = chunk['chunk'].get('key_concepts', [])
            all_concepts.extend(concepts)
        
        # Remove duplicates and create nodes
        unique_concepts = list(set(all_concepts))[:15]  # Limit to 15 concepts
        
        for i, concept in enumerate(unique_concepts):
            nodes.append({
                'id': i,
                'label': concept,
                'size': random.randint(10, 30),
                'color': f'hsl({random.randint(0, 360)}, 70%, 60%)'
            })
        
        # Create edges (relationships between concepts)
        for i in range(len(nodes)):
            for j in range(i+1, min(i+4, len(nodes))):  # Connect to 3 nearby concepts
                if random.random() > 0.3:  # 70% chance of connection
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': random.uniform(0.3, 1.0)
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def create_adaptive_response(self, question: str, context: str, level: str, personality: str, pdf_name: str) -> str:
        """Create adaptive response based on parameters"""
        personality_info = self.personality_types[personality]
        
        if level == 'beginner':
            response = f"""**{personality_info['name']} says:** {personality_info['greeting']}

**Simple Explanation:**
Let me break this down in easy terms! Based on "{pdf_name}", here's what I found about your question: "{question}"

**The Main Idea:**
{context[:300]}...

**In Simple Words:**
Think of it like this - imagine you're explaining this to a friend who's never heard of this topic before. The key points are really about understanding the basic concepts first.

**What This Means:**
• The document shows us some important ideas
• These ideas connect to bigger concepts
• Understanding this helps with other related topics

**Next Steps:**
Try thinking about how this connects to things you already know!"""

        elif level == 'intermediate':
            response = f"""**{personality_info['name']} here:** {personality_info['greeting']}

**Detailed Analysis:**
Great question! From "{pdf_name}", I can provide a comprehensive explanation of "{question}"

**Key Information:**
{context[:400]}...

**Understanding the Concept:**
This involves several interconnected elements that build upon fundamental principles. The relationship between these concepts demonstrates...

**Technical Aspects:**
• Primary mechanisms involved
• Supporting evidence from the text
• Implications for broader understanding

**Application:**
Consider how this principle applies to real-world scenarios and connects with related theories."""

        else:  # advanced
            response = f"""**{personality_info['name']} - Advanced Analysis:** {personality_info['greeting']}

**Comprehensive Examination:**
Analyzing "{question}" from "{pdf_name}" requires deep consideration of multiple theoretical frameworks.

**Detailed Context:**
{context[:500]}...

**Critical Analysis:**
The sophisticated interplay between these elements suggests a multifaceted approach to understanding. Consider the epistemological implications...

**Advanced Considerations:**
• Theoretical foundations and assumptions
• Methodological approaches employed
• Interdisciplinary connections and implications
• Contemporary debates and alternative perspectives

**Synthesis:**
This analysis reveals the complex nature of the subject matter and its position within the broader academic discourse."""
        
        return response
    
    def generate_follow_up_questions(self, context: str, level: str) -> List[str]:
        """Generate follow-up questions for active recall"""
        if level == 'beginner':
            questions = [
                "Can you explain this in your own words?",
                "What is the most important point here?",
                "How does this relate to something you already know?",
                "What questions do you still have about this topic?"
            ]
        elif level == 'intermediate':
            questions = [
                "How would you apply this concept to a different situation?",
                "What are the strengths and limitations of this approach?",
                "How does this connect to other topics in the field?",
                "What evidence supports this explanation?"
            ]
        else:
            questions = [
                "What are the theoretical implications of this analysis?",
                "How might alternative frameworks interpret this differently?",
                "What are the methodological considerations involved?",
                "How does this advance our understanding of the field?"
            ]
        
        return random.sample(questions, 3)  # Return 3 random questions
    
    def suggest_related_concepts(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """Suggest related concepts for deeper learning"""
        all_concepts = []
        for chunk in relevant_chunks:
            concepts = chunk['chunk'].get('key_concepts', [])
            topics = chunk['chunk'].get('topics', [])
            all_concepts.extend([(c, 'concept') for c in concepts])
            all_concepts.extend([(t, 'topic') for t in topics])
        
        # Create suggestions with relevance scores
        suggestions = []
        unique_items = list(set(all_concepts))[:8]
        
        for item, item_type in unique_items:
            suggestions.append({
                'name': item,
                'type': item_type,
                'relevance': random.uniform(0.6, 0.95),
                'description': f"Related {item_type} that builds on current learning"
            })
        
        return sorted(suggestions, key=lambda x: x['relevance'], reverse=True)
    
    def track_student_progress(self, question: str, answer_quality: str, topic: str, difficulty: str):
        """Track student learning progress"""
        if 'student_progress' not in st.session_state:
            st.session_state.student_progress = {
                'topics_studied': [],
                'questions_asked': 0,
                'difficulty_progression': [],
                'strong_areas': [],
                'weak_areas': [],
                'study_time': 0,
                'last_session': datetime.now().isoformat()
            }
        
        progress = st.session_state.student_progress
        progress['questions_asked'] += 1
        progress['topics_studied'].append(topic)
        progress['difficulty_progression'].append(difficulty)
        
        # Analyze performance
        if answer_quality == 'good':
            progress['strong_areas'].append(topic)
        else:
            progress['weak_areas'].append(topic)
        
        progress['last_session'] = datetime.now().isoformat()
    
    def create_revision_schedule(self, weak_topics: List[str]) -> List[Dict]:
        """Create spaced repetition schedule for difficult topics"""
        schedule = []
        base_date = datetime.now()
        
        intervals = [1, 3, 7, 14, 30]  # Spaced repetition intervals in days
        
        for topic in weak_topics[:5]:  # Top 5 weak topics
            for i, interval in enumerate(intervals):
                review_date = base_date + timedelta(days=interval)
                schedule.append({
                    'topic': topic,
                    'review_date': review_date.strftime('%Y-%m-%d'),
                    'interval_days': interval,
                    'review_type': ['quick_review', 'practice_questions', 'deep_study', 'application', 'assessment'][i]
                })
        
        return sorted(schedule, key=lambda x: x['review_date'])
    
    # [Previous exam methods remain the same, but with enhanced features]
    def generate_exam_questions(self, chunks: List[Dict], num_questions: int = 10, difficulty: str = "medium") -> List[Dict]:
        """Generate exam questions with enhanced features"""
        questions = []
        selected_chunks = random.sample(chunks, min(len(chunks), num_questions * 2))
        
        for i in range(min(num_questions, len(selected_chunks))):
            chunk = selected_chunks[i]
            question_type = random.choice(self.question_types)
            
            if question_type == 'multiple_choice':
                question = self.create_enhanced_mcq(chunk, difficulty)
            elif question_type == 'true_false':
                question = self.create_enhanced_tf(chunk, difficulty)
            elif question_type == 'short_answer':
                question = self.create_enhanced_sa(chunk, difficulty)
            else:
                question = self.create_enhanced_fill_blank(chunk, difficulty)
            
            if question:
                questions.append(question)
        
        return questions[:num_questions]
    
    def create_enhanced_mcq(self, chunk: Dict, difficulty: str) -> Dict:
        """Enhanced multiple choice question creation"""
        content = chunk['text']
        concepts = chunk.get('key_concepts', [])
        
        question_text = f"Based on the concepts {', '.join(concepts[:2])}, which statement is most accurate?"
        
        correct_answer = "This aligns with the fundamental principles described in the text"
        distractors = [
            "This contradicts the main evidence presented",
            "This represents an oversimplification of the concept",
            "This extends beyond the scope of the given information"
        ]
        
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        return {
            'type': 'multiple_choice',
            'question': question_text,
            'options': options,
            'correct_answer': correct_answer,
            'explanation': f"Based on the key concepts {', '.join(concepts[:2])}, this answer best reflects the content.",
            'difficulty': difficulty,
            'chunk_id': chunk['id'],
            'concepts_tested': concepts[:2],
            'bloom_level': 'application' if difficulty == 'hard' else 'comprehension'
        }
    
    def create_enhanced_tf(self, chunk: Dict, difficulty: str) -> Dict:
        """Enhanced true/false question"""
        concepts = chunk.get('key_concepts', ['the main concept'])
        is_true = random.choice([True, False])
        
        if is_true:
            statement = f"The text supports the idea that {concepts[0] if concepts else 'the main concept'} is fundamental to understanding this topic."
        else:
            statement = f"The text indicates that {concepts[0] if concepts else 'the main concept'} contradicts established principles."
        
        return {
            'type': 'true_false',
            'question': statement,
            'correct_answer': is_true,
            'explanation': f"This statement is {'true' if is_true else 'false'} based on the content analysis.",
            'difficulty': difficulty,
            'chunk_id': chunk['id'],
            'concepts_tested': concepts[:1],
            'bloom_level': 'analysis' if difficulty == 'hard' else 'knowledge'
        }
    
    def create_enhanced_sa(self, chunk: Dict, difficulty: str) -> Dict:
        """Enhanced short answer question"""
        concepts = chunk.get('key_concepts', [])
        topics = chunk.get('topics', [])
        
        question_text = f"Explain how {concepts[0] if concepts else 'the main concept'} relates to {topics[0] if topics else 'the broader topic'} discussed in this section."
        
        return {
            'type': 'short_answer',
            'question': question_text,
            'sample_answer': f"The relationship involves understanding how {concepts[0] if concepts else 'key elements'} function within the broader context...",
            'explanation': "A complete answer should demonstrate understanding of the connections between concepts.",
            'difficulty': difficulty,
            'chunk_id': chunk['id'],
            'concepts_tested': concepts[:2],
            'bloom_level': 'synthesis' if difficulty == 'hard' else 'comprehension'
        }
    
    def create_enhanced_fill_blank(self, chunk: Dict, difficulty: str) -> Dict:
        """Enhanced fill-in-the-blank question"""
        concepts = chunk.get('key_concepts', [])
        if not concepts:
            return None
            
        concept = concepts[0]
        question_text = f"The fundamental principle of _______ is essential for understanding this topic area."
        
        return {
            'type': 'fill_blank',
            'question': question_text,
            'correct_answer': concept,
            'explanation': f"The term '{concept}' is a key concept identified in this section.",
            'difficulty': difficulty,
            'chunk_id': chunk['id'],
            'concepts_tested': [concept],
            'bloom_level': 'knowledge'
        }
    
    def grade_exam_enhanced(self, questions: List[Dict], user_answers: Dict) -> Dict:
        """Enhanced exam grading with detailed analytics"""
        total_questions = len(questions)
        correct_answers = 0
        concept_performance = {}
        bloom_performance = {}
        feedback = []
        
        for i, question in enumerate(questions):
            user_answer = user_answers.get(f"q_{i}", "")
            
            # Grade based on question type
            if question['type'] == 'multiple_choice':
                is_correct = user_answer == question['correct_answer']
            elif question['type'] == 'true_false':
                is_correct = str(user_answer).lower() == str(question['correct_answer']).lower()
            elif question['type'] == 'fill_blank':
                is_correct = user_answer.lower().strip() in question['correct_answer'].lower()
            else:  # short_answer
                is_correct = len(user_answer.strip()) > 20 and any(concept.lower() in user_answer.lower() 
                                                                 for concept in question.get('concepts_tested', []))
            
            if is_correct:
                correct_answers += 1
            
            # Track concept performance
            for concept in question.get('concepts_tested', []):
                if concept not in concept_performance:
                    concept_performance[concept] = {'correct': 0, 'total': 0}
                concept_performance[concept]['total'] += 1
                if is_correct:
                    concept_performance[concept]['correct'] += 1
            
            # Track Bloom's taxonomy performance
            bloom_level = question.get('bloom_level', 'knowledge')
            if bloom_level not in bloom_performance:
                bloom_performance[bloom_level] = {'correct': 0, 'total': 0}
            bloom_performance[bloom_level]['total'] += 1
            if is_correct:
                bloom_performance[bloom_level]['correct'] += 1
            
            feedback.append({
                'question_num': i + 1,
                'question': question['question'],
                'user_answer': user_answer,
                'correct_answer': question.get('correct_answer', question.get('sample_answer', '')),
                'is_correct': is_correct,
                'explanation': question['explanation'],
                'type': question['type'],
                'concepts_tested': question.get('concepts_tested', []),
                'bloom_level': bloom_level,
                'difficulty': question.get('difficulty', 'medium')
            })
        
        score = (correct_answers / total_questions) * 100
        
        # Enhanced grading with multiple metrics
        letter_grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 60 else 'F'
        
        return {
            'score': score,
            'grade': letter_grade,
            'correct': correct_answers,
            'total': total_questions,
            'feedback': feedback,
            'concept_performance': concept_performance,
            'bloom_performance': bloom_performance,
            'recommendations': self.generate_study_recommendations(concept_performance, bloom_performance)
        }
    
    def generate_study_recommendations(self, concept_perf: Dict, bloom_perf: Dict) -> List[str]:
        """Generate personalized study recommendations"""
        recommendations = []
        
        # Analyze weak concepts
        weak_concepts = [concept for concept, perf in concept_perf.items() 
                        if perf['correct'] / perf['total'] < 0.7]
        
        if weak_concepts:
            recommendations.append(f"📚 Focus on reviewing: {', '.join(weak_concepts[:3])}")
        
        # Analyze cognitive levels
        weak_bloom = [level for level, perf in bloom_perf.items() 
                     if perf['correct'] / perf['total'] < 0.6]
        
        if 'analysis' in weak_bloom:
            recommendations.append("🔍 Practice analytical thinking with case studies")
        if 'application' in weak_bloom:
            recommendations.append("🛠️ Work on applying concepts to new scenarios")
        if 'synthesis' in weak_bloom:
            recommendations.append("🧩 Practice combining ideas from different topics")
        
        return recommendations[:4]  # Top 4 recommendations

def load_sample_pdfs():
    """Enhanced sample PDFs with more metadata"""
    return {
        "Advanced Physics - Quantum Mechanics": {
            "description": "Quantum theory, wave-particle duality, and quantum applications",
            "difficulty": "advanced",
            "topics": ["physics", "quantum"],
            "sample_questions": [
                "Explain the concept of wave-particle duality with examples",
                "What is quantum entanglement and how does it work?",
                "Describe the uncertainty principle and its implications"
            ]
        },
        "Molecular Biology Fundamentals": {
            "description": "DNA structure, protein synthesis, and cellular processes",
            "difficulty": "intermediate",
            "topics": ["biology", "molecular"],
            "sample_questions": [
                "Describe the process of DNA replication in detail",
                "How does transcription differ from translation?",
                "Explain the role of enzymes in cellular metabolism"
            ]
        },
        "Calculus and Analysis": {
            "description": "Limits, derivatives, integrals, and real analysis",
            "difficulty": "advanced",
            "topics": ["mathematics", "calculus"],
            "sample_questions": [
                "Prove the fundamental theorem of calculus",
                "What is the geometric interpretation of derivatives?",
                "How do you evaluate improper integrals?"
            ]
        }
    }

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'study_system': AdvancedStudyMateSystem(),
        'current_pdf_data': None,
        'chat_history': [],
        'current_exam': None,
        'exam_answers': {},
        'exam_results': None,
        'exam_history': [],
        'student_progress': {
            'topics_studied': [],
            'questions_asked': 0,
            'difficulty_progression': [],
            'strong_areas': [],
            'weak_areas': [],
            'study_time': 0,
            'learning_level': 'intermediate',
            'preferred_personality': 'friendly',
            'last_session': datetime.now().isoformat()
        },
        'revision_schedule': [],
        'collaborative_sessions': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    """Main application function"""
    initialize_session_state()
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 StudyMate - Advanced AI Learning Companion</h1>
        <p>Personalized learning with Smart Adaptation, Visual Concept Mapping & Active Recall</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Smart Q&A", "📝 Adaptive Exams", "📊 Progress Analytics", 
        "🔄 Active Recall", "👥 Collaboration Hub"
    ])
    
    with tab1:
        smart_qa_mode()
    
    with tab2:
        adaptive_exam_mode()
    
    with tab3:
        progress_analytics()
    
    with tab4:
        active_recall_mode()
    
    with tab5:
        collaboration_hub()

def create_sidebar():
    """Enhanced sidebar with all features"""
    with st.sidebar:
        st.header("📄 Document Management")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=['pdf'],
    help="Upload any academic PDF for intelligent analysis",
    key="pdf_uploader_sidebar"
)

        
        if uploaded_file is not None:
            if st.button("🚀 Process PDF with AI", type="primary"):
                with st.spinner("🔍 Analyzing document with advanced AI..."):
                    text = st.session_state.study_system.extract_text_from_pdf(uploaded_file)
                    if text:
                        chunks = st.session_state.study_system.chunk_text(text)
                        embeddings = st.session_state.study_system.create_embeddings(chunks)
                        
                        st.session_state.current_pdf_data = {
                            'name': uploaded_file.name,
                            'text': text,
                            'chunks': chunks,
                            'embeddings': embeddings,
                            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'total_concepts': sum(len(chunk.get('key_concepts', [])) for chunk in chunks),
                            'complexity_distribution': {
                                level: sum(1 for chunk in chunks if chunk.get('complexity') == level)
                                for level in ['beginner', 'intermediate', 'advanced']
                            }
                        }
                        
                        st.success("✅ Document processed with AI enhancement!")
                        st.info(f"📊 Found {len(chunks)} sections, {st.session_state.current_pdf_data['total_concepts']} key concepts")
        
        # Student Profile Configuration
        st.header("👤 Your Learning Profile")
        
        current_level = st.session_state.student_progress['learning_level']
        learning_level = st.selectbox(
            "Academic Level",
            ['beginner', 'intermediate', 'advanced'],
            index=['beginner', 'intermediate', 'advanced'].index(current_level),
            help="Adjust explanations to your level"
        )
        st.session_state.student_progress['learning_level'] = learning_level
        
        current_personality = st.session_state.student_progress['preferred_personality']
        tutor_personality = st.selectbox(
            "AI Tutor Personality",
            list(st.session_state.study_system.personality_types.keys()),
            index=list(st.session_state.study_system.personality_types.keys()).index(current_personality),
            format_func=lambda x: st.session_state.study_system.personality_types[x]['name'],
            help="Choose your preferred AI tutor style"
        )
        st.session_state.student_progress['preferred_personality'] = tutor_personality
        
        # Display current tutor personality
        personality_info = st.session_state.study_system.personality_types[tutor_personality]
        st.markdown(f"""
        <div class="tutor-personality tutor-{tutor_personality}">
            <strong>{personality_info['name']}</strong><br>
            <small>{personality_info['style']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample PDFs with enhanced metadata
        st.header("📖 Sample Documents")
        sample_pdfs = load_sample_pdfs()
        
        for pdf_name, info in sample_pdfs.items():
            if st.button(f"📄 {pdf_name}", key=f"sample_{pdf_name}"):
                # Create enhanced sample PDF data
                sample_chunks = [
                    {
                        'id': 0,
                        'text': f"Advanced content for {pdf_name}. This comprehensive material covers {', '.join(info['topics'])} at {info['difficulty']} level.",
                        'length': 200,
                        'complexity': info['difficulty'],
                        'key_concepts': info['topics'] + ['analysis', 'theory', 'application'],
                        'topics': info['topics']
                    }
                ]
                
                st.session_state.current_pdf_data = {
                    'name': f"{pdf_name}.pdf",
                    'text': f"Sample content for {pdf_name}",
                    'chunks': sample_chunks,
                    'embeddings': np.random.rand(1, 384),
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'is_sample': True,
                    'difficulty': info['difficulty'],
                    'topics': info['topics'],
                    'total_concepts': len(sample_chunks[0]['key_concepts']),
                    'complexity_distribution': {info['difficulty']: 1, 'beginner': 0, 'intermediate': 0, 'advanced': 0}
                }
                st.success(f"✅ Loaded sample: {pdf_name}")
        
        # Current document info with enhanced details
        if st.session_state.current_pdf_data:
            st.header("📋 Document Analysis")
            doc_data = st.session_state.current_pdf_data
            st.info(f"**📄 Document:** {doc_data['name']}")
            st.info(f"**🕐 Processed:** {doc_data['processed_at']}")
            st.info(f"**🧠 Concepts:** {doc_data.get('total_concepts', 0)}")
            
            if 'complexity_distribution' in doc_data:
                st.write("**Complexity Distribution:**")
                for level, count in doc_data['complexity_distribution'].items():
                    if count > 0:
                        st.write(f"• {level.title()}: {count} sections")
            
            if st.button("🗑️ Clear Document"):
                keys_to_clear = ['current_pdf_data', 'chat_history', 'current_exam', 'exam_answers', 'exam_results']
                for key in keys_to_clear:
                    st.session_state[key] = None if key == 'current_pdf_data' else [] if 'history' in key else {}
                st.rerun()

def smart_qa_mode():
    """Enhanced Q&A mode with all advanced features"""
    create_sidebar()
    
    st.markdown("### 🎯 Intelligent Question Answering with Visual Concept Mapping")
    
    if not st.session_state.current_pdf_data:
        st.info("👆 Please upload a PDF document or select a sample to begin intelligent learning!")
        
        # Display welcome features
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 🌟 Smart Learning Features
            - **Adaptive Explanations** - Content adjusts to your level
            - **Visual Concept Maps** - See how ideas connect
            - **Confidence Scoring** - Know how certain the AI is
            - **Multi-modal Support** - Handles text, diagrams, equations
            """)
        
        with col2:
            st.markdown("""
            #### 🚀 Advanced Capabilities  
            - **Active Recall Questions** - Reinforces learning
            - **Related Concept Suggestions** - Expand your knowledge
            - **Progress Tracking** - Monitor your improvement
            - **Personalized AI Tutors** - Choose your learning style
            """)
        return
    
    # Enhanced question interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Intelligent Tutor")
        
        # Enhanced question input with suggestions
        question = st.text_area(
            "What would you like to explore today?",
            height=100,
            placeholder="Example: Explain quantum entanglement and show me how it connects to other quantum concepts"
        )
        
        # Quick question buttons based on document
        if st.session_state.current_pdf_data.get('is_sample'):
            sample_info = None
            for name, info in load_sample_pdfs().items():
                if name in st.session_state.current_pdf_data['name']:
                    sample_info = info
                    break
            
            if sample_info:
                st.write("💡 **Quick Questions:**")
                cols = st.columns(len(sample_info['sample_questions']))
                for i, q in enumerate(sample_info['sample_questions']):
                    with cols[i]:
                        if st.button(f"❓ {q}", key=f"quick_{i}"):
                            question = q
        
        # Advanced answer generation
        if st.button("🧠 Get Intelligent Answer", type="primary", disabled=not question):
            if question:
                with st.spinner("🔍 Analyzing with advanced AI..."):
                    relevant_chunks = st.session_state.study_system.find_relevant_chunks(
                        question,
                        st.session_state.current_pdf_data['chunks'],
                        st.session_state.current_pdf_data['embeddings']
                    )
                    
                    # Generate comprehensive answer
                    response_data = st.session_state.study_system.generate_adaptive_answer(
                        question,
                        relevant_chunks,
                        st.session_state.student_progress['learning_level'],
                        st.session_state.student_progress['preferred_personality'],
                        st.session_state.current_pdf_data['name']
                    )
                    
                    # Track progress
                    st.session_state.study_system.track_student_progress(
                        question, 'good', 'general', 
                        st.session_state.student_progress['learning_level']
                    )
                    
                    # Add to enhanced chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'response_data': response_data,
                        'timestamp': datetime.now().strftime("%H:%M"),
                        'session_id': len(st.session_state.chat_history) + 1
                    })
                    
                    st.success("✅ Intelligent analysis complete!")
                    st.rerun()
        
        # Display enhanced chat history
        if st.session_state.chat_history:
            st.header("📚 Your Learning Journey")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
                with st.expander(f"💬 Session {chat['session_id']}: {chat['question'][:50]}... ({chat['timestamp']})", expanded=i==0):
                    response_data = chat['response_data']
                    
                    # Display answer with confidence
                    st.markdown("**🤖 AI Tutor Response:**")
                    st.markdown(response_data['answer'])
                    
                    # Confidence meter
                    confidence = response_data['confidence']
                    confidence_color = 'high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low'
                    st.markdown(f"""
                    **🎯 Answer Confidence:** <span class="confidence-{confidence_color}">
                    {confidence:.1%} ({confidence_color.upper()})
                    </span>
                    """, unsafe_allow_html=True)
                    
                    # Visual concept map
                    if response_data['concept_map']['nodes']:
                        st.markdown("**🗺️ Concept Relationships:**")
                        display_concept_map(response_data['concept_map'])
                    
                    # Active recall questions
                    if response_data['follow_up_questions']:
                        st.markdown("**🧠 Test Your Understanding:**")
                        for fq in response_data['follow_up_questions']:
                            st.write(f"• {fq}")
                    
                    # Related concepts
                    if response_data['related_concepts']:
                        st.markdown("**🔗 Explore Related Concepts:**")
                        for concept in response_data['related_concepts'][:4]:
                            st.write(f"• **{concept['name']}** ({concept['relevance']:.0%} relevant) - {concept['description']}")
    
    with col2:
        st.header("🎯 Learning Assistant")
        
        # Real-time progress metrics
        progress = st.session_state.student_progress
        st.metric("Questions Asked", progress['questions_asked'])
        st.metric("Topics Explored", len(set(progress['topics_studied'])))
        st.metric("Current Level", progress['learning_level'].title())
        
        # Learning tips based on progress
        st.markdown("### 💡 Smart Learning Tips")
        
        if progress['questions_asked'] < 3:
            st.info("🌱 Just getting started! Try asking follow-up questions to deepen understanding.")
        elif len(progress['weak_areas']) > len(progress['strong_areas']):
            st.warning("📖 Focus on reviewing concepts you find challenging. Use the Active Recall tab!")
        else:
            st.success("🌟 Great progress! Try exploring advanced topics or take a practice exam.")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🎯 Generate Practice Quiz"):
            st.session_state.quick_action = 'generate_quiz'
        if st.button("📊 View My Progress"):
            st.session_state.quick_action = 'view_progress'
        if st.button("🔄 Schedule Review"):
            st.session_state.quick_action = 'schedule_review'

def display_concept_map(concept_map_data):
    """Display interactive concept map visualization"""
    nodes = concept_map_data['nodes']
    edges = concept_map_data['edges']
    
    if not nodes:
        st.write("No concept relationships found.")
        return
    
    # Create network visualization using Plotly
    edge_trace = []
    node_trace = []
    
    # Process edges
    for edge in edges:
        source_node = nodes[edge['source']]
        target_node = nodes[edge['target']]
        
        edge_trace.append(
            go.Scatter(
                x=[source_node['id'], target_node['id']],
                y=[0, 1],  # Simple positioning
                mode='lines',
                line=dict(width=edge['weight'] * 3, color='lightblue'),
                showlegend=False
            )
        )
    
    # Process nodes
    for i, node in enumerate(nodes):
        node_trace.append(
            go.Scatter(
                x=[i % 5],  # Arrange in grid
                y=[i // 5],
                mode='markers+text',
                marker=dict(size=node['size'], color=node['color']),
                text=node['label'],
                textposition="middle center",
                showlegend=False
            )
        )
    
    # Create figure
    fig = go.Figure(data=node_trace + edge_trace)
    fig.update_layout(
        title="🗺️ Concept Map - How Ideas Connect",
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def adaptive_exam_mode():
    """Enhanced exam mode with advanced analytics"""
    create_sidebar()
    
    st.markdown("### 📝 Adaptive Examination System")
    
    if not st.session_state.current_pdf_data:
        st.warning("Please upload a PDF document first to generate intelligent exam questions.")
        return
    
    # Enhanced exam configuration
    st.header("🎯 Create Your Personalized Exam")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_questions = st.selectbox("Questions", [5, 10, 15, 20], index=1)
    with col2:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    with col3:
        time_limit = st.selectbox("Time (min)", [10, 15, 30, 45, 60], index=2)
    with col4:
        focus_area = st.selectbox("Focus", ["all_topics", "weak_areas", "strong_areas"], 
                                 format_func=lambda x: x.replace('_', ' ').title())
    
    # Bloom's taxonomy selection
    st.subheader("🎓 Cognitive Skills to Test")
    bloom_levels = st.multiselect(
        "Select thinking levels (Bloom's Taxonomy)",
        ['knowledge', 'comprehension', 'application', 'analysis', 'synthesis', 'evaluation'],
        default=['comprehension', 'application'],
        help="Choose the types of thinking skills to assess"
    )
    
    # Generate enhanced exam
    if st.button("🚀 Generate Intelligent Exam", type="primary"):
        with st.spinner("🧠 Creating personalized exam questions..."):
            # Filter chunks based on focus area
            chunks = st.session_state.current_pdf_data['chunks']
            if focus_area == "weak_areas" and st.session_state.student_progress['weak_areas']:
                # Filter chunks related to weak areas (simplified)
                chunks = chunks[:len(chunks)//2]
            elif focus_area == "strong_areas" and st.session_state.student_progress['strong_areas']:
                chunks = chunks[len(chunks)//2:]
            
            questions = st.session_state.study_system.generate_exam_questions(
                chunks, num_questions=num_questions, difficulty=difficulty
            )
            
            st.session_state.current_exam = {
                'questions': questions,
                'settings': {
                    'num_questions': num_questions,
                    'difficulty': difficulty,
                    'time_limit': time_limit,
                    'focus_area': focus_area,
                    'bloom_levels': bloom_levels,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'adaptive_level': st.session_state.student_progress['learning_level']
                }
            }
            st.session_state.exam_answers = {}
            st.session_state.exam_results = None
            
        st.success(f"✅ Generated {len(questions)} personalized questions!")
        st.rerun()
    
    # Display current exam or results
    if st.session_state.current_exam and not st.session_state.exam_results:
        display_enhanced_exam()
    elif st.session_state.exam_results:
        display_enhanced_exam_results()
    
    # Enhanced exam history with analytics
    display_exam_analytics()

def display_enhanced_exam():
    """Display exam with enhanced features"""
    exam = st.session_state.current_exam
    questions = exam['questions']
    settings = exam['settings']
    
    st.header(f"📝 Adaptive Exam - {settings['difficulty'].title()} Level")
    
    # Exam info bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Questions:** {len(questions)}")
    with col2:
        st.info(f"**Time Limit:** {settings['time_limit']} min")
    with col3:
        st.info(f"**Focus:** {settings['focus_area'].replace('_', ' ').title()}")
    with col4:
        st.info(f"**Level:** {settings['adaptive_level'].title()}")
    
    # Timer simulation (in real app, would be live)
    progress_bar = st.progress(0)
    st.write("⏱️ Time remaining: [Timer would be active during real exam]")
    
    # Enhanced exam form
    with st.form("enhanced_exam_form"):
        st.markdown("---")
        
        for i, question in enumerate(questions):
            # Question header with metadata
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"Question {i+1}")
            with col2:
                st.caption(f"Type: {question['type'].replace('_', ' ').title()}")
                if 'bloom_level' in question:
                    st.caption(f"Skill: {question['bloom_level'].title()}")
            
            st.write(question['question'])
            
            # Enhanced input based on question type
            if question['type'] == 'multiple_choice':
                answer = st.radio(
                    f"Select your answer:",
                    question['options'],
                    key=f"q_{i}",
                    index=None
                )
                
            elif question['type'] == 'true_false':
                answer = st.radio(
                    f"Select your answer:",
                    [True, False],
                    key=f"q_{i}",
                    index=None,
                    format_func=lambda x: "True" if x else "False"
                )
                
            elif question['type'] == 'fill_blank':
                answer = st.text_input(
                    f"Fill in the blank:",
                    key=f"q_{i}",
                    placeholder="Enter your answer here..."
                )
                
            else:  # short_answer
                answer = st.text_area(
                    f"Provide your detailed answer:",
                    key=f"q_{i}",
                    height=120,
                    placeholder="Write your comprehensive answer here...",
                    help="Aim for a detailed response that demonstrates your understanding"
                )
            
            # Show concepts being tested
            if 'concepts_tested' in question and question['concepts_tested']:
                st.caption(f"🎯 Testing: {', '.join(question['concepts_tested'])}")
            
            st.markdown("---")
        
        # Enhanced submit section
        st.subheader("📋 Submit Your Exam")
        confidence_check = st.checkbox("I am confident in my answers and ready to submit")
        
        submitted = st.form_submit_button(
            "🎯 Submit Exam for Analysis", 
            type="primary",
            disabled=not confidence_check
        )
        
        if submitted:
            # Collect and grade answers
            answers = {}
            for i in range(len(questions)):
                answers[f"q_{i}"] = st.session_state.get(f"q_{i}", "")
            
            # Enhanced grading
            results = st.session_state.study_system.grade_exam_enhanced(questions, answers)
            
            st.session_state.exam_results = results
            st.session_state.exam_answers = answers
            
            # Add to enhanced history
            exam_record = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'score': results['score'],
                'grade': results['grade'],
                'correct': results['correct'],
                'total': results['total'],
                'difficulty': settings['difficulty'],
                'focus_area': settings['focus_area'],
                'adaptive_level': settings['adaptive_level'],
                'concept_performance': results['concept_performance'],
                'bloom_performance': results['bloom_performance'],
                'recommendations': results['recommendations']
            }
            st.session_state.exam_history.append(exam_record)
            
            st.rerun()

def display_enhanced_exam_results():
    """Display comprehensive exam results"""
    results = st.session_state.exam_results
    
    st.header("🎉 Comprehensive Exam Analysis")
    
    # Enhanced score display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{results['score']:.1f}%")
    with col2:
        st.metric("Letter Grade", results['grade'])
    with col3:
        st.metric("Questions Correct", f"{results['correct']}/{results['total']}")
    with col4:
        accuracy = (results['correct'] / results['total']) * 100
        st.metric("Accuracy Rate", f"{accuracy:.0f}%")
    
    # Performance visualization
    if results.get('concept_performance'):
        st.subheader("📊 Concept Mastery Analysis")
        
        concept_data = []
        for concept, perf in results['concept_performance'].items():
            accuracy = (perf['correct'] / perf['total']) * 100
            concept_data.append({'Concept': concept, 'Accuracy': accuracy, 'Questions': perf['total']})
        
        if concept_data:
            # Create concept performance chart
            fig = px.bar(
                concept_data, 
                x='Concept', 
                y='Accuracy',
                title='Concept Understanding Levels',
                color='Accuracy',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Bloom's taxonomy analysis
    if results.get('bloom_performance'):
        st.subheader("🎓 Cognitive Skills Assessment")
        
        bloom_data = []
        for level, perf in results['bloom_performance'].items():
            accuracy = (perf['correct'] / perf['total']) * 100
            bloom_data.append({'Thinking Level': level.title(), 'Performance': accuracy})
        
        if bloom_data:
            fig = px.pie(
                bloom_data,
                values='Performance',
                names='Thinking Level',
                title='Cognitive Skills Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Personalized recommendations
    st.subheader("🎯 Personalized Study Recommendations")
    if results.get('recommendations'):
        for rec in results['recommendations']:
            st.info(rec)
    else:
        if results['score'] >= 85:
            st.success("🌟 Excellent performance! Consider exploring advanced topics or helping others.")
        elif results['score'] >= 70:
            st.info("👍 Good work! Focus on the areas highlighted above for improvement.")
        else:
            st.warning("📚 More study needed. Consider using the Active Recall feature and reviewing key concepts.")
    
    # Detailed question feedback
    st.subheader("📋 Question-by-Question Analysis")
    
    for feedback in results['feedback']:
        with st.expander(
            f"Q{feedback['question_num']}: {'✅ Correct' if feedback['is_correct'] else '❌ Incorrect'} "
            f"({feedback['difficulty'].title()} - {feedback['bloom_level'].title()})"
        ):
            st.write(f"**Question:** {feedback['question']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Your Answer:** {feedback['user_answer']}")
            with col2:
                st.write(f"**Correct Answer:** {feedback['correct_answer']}")
            
            st.write(f"**Explanation:** {feedback['explanation']}")
            
            if feedback['concepts_tested']:
                st.write(f"**Concepts Tested:** {', '.join(feedback['concepts_tested'])}")
            
            if feedback['is_correct']:
                st.success("✅ Well done! Your understanding is solid.")
            else:
                st.error("❌ Review this concept. Consider using the Q&A mode for deeper understanding.")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 New Exam"):
            st.session_state.current_exam = None
            st.session_state.exam_answers = {}
            st.session_state.exam_results = None
            st.rerun()
    
    with col2:
        if st.button("📚 Study Weak Areas"):
            # Set up active recall for weak concepts
            if results.get('concept_performance'):
                weak_concepts = [concept for concept, perf in results['concept_performance'].items()
                               if perf['correct'] / perf['total'] < 0.7]
                st.session_state.active_recall_topics = weak_concepts
            st.session_state.active_tab = 'active_recall'
    
    with col3:
        if st.button("📊 View Progress Analytics"):
            st.session_state.active_tab = 'progress'

def progress_analytics():
    """Comprehensive progress analytics dashboard"""
    st.markdown("### 📊 Your Learning Analytics Dashboard")
    
    progress = st.session_state.student_progress
    exam_history = st.session_state.exam_history
    
    if not exam_history and progress['questions_asked'] == 0:
        st.info("📈 Start learning to see your progress analytics here!")
        
        # Show sample analytics
        st.markdown("#### 📊 Sample Analytics (What you'll see)")
        
        # Sample charts
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Score': [75, 82, 88],
            'Questions': [15, 20, 18]
        }
        
        fig = px.line(sample_data, x='Date', y='Score', title='Score Progress Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Overall statistics
    st.header("📈 Overall Learning Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", progress['questions_asked'])
    with col2:
        st.metric("Topics Studied", len(set(progress['topics_studied'])))
    with col3:
        st.metric("Exams Taken", len(exam_history))
    with col4:
        avg_score = np.mean([exam['score'] for exam in exam_history]) if exam_history else 0
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    # Progress over time
    if exam_history:
        st.header("📊 Performance Trends")
        
        # Score progression
        exam_dates = [datetime.strptime(exam['date'], "%Y-%m-%d %H:%M:%S").strftime("%m/%d") for exam in exam_history]
        exam_scores = [exam['score'] for exam in exam_history]
        
        fig = px.line(
            x=exam_dates, 
            y=exam_scores,
            title='📈 Score Progression Over Time',
            labels={'x': 'Date', 'y': 'Score (%)'}
        )
        fig.update_traces(line_color='#667eea', line_width=3)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance by difficulty
        difficulty_stats = {}
        for exam in exam_history:
            diff = exam['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = []
            difficulty_stats[diff].append(exam['score'])
        
        if len(difficulty_stats) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                diff_data = []
                for diff, scores in difficulty_stats.items():
                    diff_data.append({
                        'Difficulty': diff.title(),
                        'Average Score': np.mean(scores),
                        'Exams Taken': len(scores)
                    })
                
                fig = px.bar(
                    diff_data,
                    x='Difficulty',
                    y='Average Score',
                    title='Performance by Difficulty Level',
                    color='Average Score',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Strong vs weak areas
                strong_count = len(progress['strong_areas'])
                weak_count = len(progress['weak_areas'])
                
                if strong_count + weak_count > 0:
                    fig = px.pie(
                        values=[strong_count, weak_count],
                        names=['Strong Areas', 'Areas to Improve'],
                        title='Learning Balance',
                        color_discrete_map={
                            'Strong Areas': '#28a745',
                            'Areas to Improve': '#ffc107'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Learning insights
    st.header("🧠 Learning Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Your Strengths")
        if progress['strong_areas']:
            strong_topics = list(set(progress['strong_areas']))[:5]
            for topic in strong_topics:
                st.success(f"✅ {topic.title()}")
        else:
            st.info("Keep learning to discover your strengths!")
    
    with col2:
        st.subheader("📚 Areas to Focus On")
        if progress['weak_areas']:
            weak_topics = list(set(progress['weak_areas']))[:5]
            for topic in weak_topics:
                st.warning(f"📖 {topic.title()}")
        else:
            st.success("No weak areas identified yet!")
    
    # Study recommendations
    st.header("🎯 Personalized Study Plan")
    
    if exam_history:
        latest_exam = exam_history[-1]
        
        if latest_exam['score'] >= 90:
            st.success("🌟 **Excellent Progress!** Consider exploring advanced topics or mentoring others.")
        elif latest_exam['score'] >= 80:
            st.info("👍 **Good Progress!** Focus on consistency and challenging yourself with harder material.")
        elif latest_exam['score'] >= 70:
            st.warning("📖 **Steady Improvement Needed.** Use active recall and spaced repetition for key concepts.")
        else:
            st.error("📚 **Intensive Study Required.** Consider one-on-one tutoring and fundamental concept review.")
        
        # Specific recommendations
        if 'recommendations' in latest_exam:
            st.write("**Specific Recommendations from Last Exam:**")
            for rec in latest_exam['recommendations']:
                st.write(f"• {rec}")
    
    # Learning streak and habits
    st.header("🔥 Learning Habits")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate study streak (simplified)
        study_days = len(set(exam['date'][:10] for exam in exam_history)) if exam_history else 0
        st.metric("Study Days", study_days)
    
    with col2:
        total_questions = progress['questions_asked'] + sum(exam['total'] for exam in exam_history)
        st.metric("Total Questions", total_questions)
    
    with col3:
        if exam_history:
            consistency = len(exam_history) / max(study_days, 1)
            st.metric("Study Consistency", f"{consistency:.1f}")

def active_recall_mode():
    """Active recall and spaced repetition system"""
    st.markdown("### 🔄 Active Recall & Spaced Repetition")
    
    if not st.session_state.current_pdf_data:
        st.info("Upload a document first to generate active recall questions!")
        return
    
    # Initialize active recall data
    if 'active_recall_data' not in st.session_state:
        st.session_state.active_recall_data = {
            'current_questions': [],
            'completed_questions': [],
            'difficulty_ratings': {},
            'review_schedule': []
        }
    
    tab1, tab2, tab3 = st.tabs(["🧠 Practice Session", "📅 Review Schedule", "📊 Recall Analytics"])
    
    with tab1:
        st.header("🎯 Active Recall Practice")
        
        # Generate practice questions if none exist
        if not st.session_state.active_recall_data['current_questions']:
            if st.button("🚀 Generate Practice Questions", type="primary"):
                with st.spinner("Creating active recall questions..."):
                    questions = []
                    chunks = st.session_state.current_pdf_data['chunks']
                    
                    # Generate different types of recall questions
                    for chunk in random.sample(chunks, min(5, len(chunks))):
                        concepts = chunk.get('key_concepts', [])
                        if concepts:
                            questions.extend([
                                {
                                    'type': 'concept_explanation',
                                    'question': f"Explain the concept of '{concepts[0]}' in your own words.",
                                    'chunk_id': chunk['id'],
                                    'difficulty': 'medium',
                                    'created': datetime.now().isoformat()
                                },
                                {
                                    'type': 'application',
                                    'question': f"How would you apply '{concepts[0]}' to solve a real-world problem?",
                                    'chunk_id': chunk['id'],
                                    'difficulty': 'hard',
                                    'created': datetime.now().isoformat()
                                }
                            ])
                    
                    st.session_state.active_recall_data['current_questions'] = questions[:8]
                    st.success(f"✅ Generated {len(st.session_state.active_recall_data['current_questions'])} practice questions!")
                    st.rerun()
        
        # Display current practice session
        current_questions = st.session_state.active_recall_data['current_questions']
        if current_questions:
            st.info(f"📚 Practice Session: {len(current_questions)} questions remaining")
            
            # Show current question
            if current_questions:
                current_q = current_questions[0]
                
                st.subheader("🤔 Active Recall Question")
                st.write(current_q['question'])
                
                # Student response
                user_response = st.text_area(
                    "Write your answer from memory (don't look it up!):",
                    height=150,
                    placeholder="Try to recall everything you know about this topic..."
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💡 Reveal Answer", disabled=not user_response):
                        # Show the answer from the document
                        relevant_chunk = next((chunk for chunk in st.session_state.current_pdf_data['chunks'] 
                                             if chunk['id'] == current_q['chunk_id']), None)
                        
                        if relevant_chunk:
                            st.success("📖 **Content from Document:**")
                            st.write(relevant_chunk['text'][:500] + "...")
                            
                            st.write("🤖 **AI Analysis:**")
                            st.write("Compare your answer with the content above. How well did you recall the key information?")
                
                with col2:
                    if st.button("✅ Next Question"):
                        # Rate difficulty and move to next
                        difficulty_rating = st.selectbox(
                            "How difficult was this question?",
                            ["easy", "medium", "hard"],
                            key="difficulty_rating"
                        )
                        
                        # Store completed question
                        completed_q = current_q.copy()
                        completed_q['user_response'] = user_response
                        completed_q['difficulty_rating'] = difficulty_rating
                        completed_q['completed_at'] = datetime.now().isoformat()
                        
                        st.session_state.active_recall_data['completed_questions'].append(completed_q)
                        st.session_state.active_recall_data['current_questions'].pop(0)
                        
                        # Schedule review based on difficulty
                        review_date = datetime.now()
                        if difficulty_rating == "easy":
                            review_date += timedelta(days=7)
                        elif difficulty_rating == "medium":
                            review_date += timedelta(days=3)
                        else:  # hard
                            review_date += timedelta(days=1)
                        
                        st.session_state.active_recall_data['review_schedule'].append({
                            'question': current_q,
                            'review_date': review_date.isoformat(),
                            'priority': 'high' if difficulty_rating == 'hard' else 'medium' if difficulty_rating == 'medium' else 'low'
                        })
                        
                        st.rerun()
        
        # Session completion
        completed = st.session_state.active_recall_data['completed_questions']
        if completed and not current_questions:
            st.success("🎉 Practice session completed!")
            
            # Session summary
            avg_difficulty = completed[-5:]  # Last 5 questions
            easy_count = sum(1 for q in avg_difficulty if q.get('difficulty_rating') == 'easy')
            medium_count = sum(1 for q in avg_difficulty if q.get('difficulty_rating') == 'medium')
            hard_count = sum(1 for q in avg_difficulty if q.get('difficulty_rating') == 'hard')
            
            st.write("📊 **Session Summary:**")
            st.write(f"• Easy: {easy_count} questions")
            st.write(f"• Medium: {medium_count} questions") 
            st.write(f"• Hard: {hard_count} questions")
            
            if hard_count > medium_count + easy_count:
                st.warning("Consider reviewing the basic concepts before your next session.")
            elif easy_count > hard_count + medium_count:
                st.success("Great recall! You're ready for more challenging material.")
    
    with tab2:
        st.header("📅 Spaced Repetition Schedule")
        
        schedule = st.session_state.active_recall_data.get('review_schedule', [])
        
        if schedule:
            # Sort by review date
            schedule.sort(key=lambda x: x['review_date'])
            
            st.subheader("🔔 Upcoming Reviews")
            
            today = datetime.now()
            due_today = []
            upcoming = []
            
            for item in schedule:
                review_date = datetime.fromisoformat(item['review_date'])
                if review_date.date() <= today.date():
                    due_today.append(item)
                else:
                    upcoming.append(item)
            
            # Due today
            if due_today:
                st.error(f"⚡ {len(due_today)} reviews due today!")
                for item in due_today[:3]:  # Show first 3
                    with st.expander(f"Due: {item['question']['question'][:50]}..."):
                        st.write(item['question']['question'])
                        st.write(f"Priority: {item['priority'].title()}")
                        if st.button(f"Review Now", key=f"review_{item['question']['chunk_id']}"):
                            # Add back to current questions
                            st.session_state.active_recall_data['current_questions'].insert(0, item['question'])
                            st.rerun()
            
            # Upcoming reviews
            if upcoming:
                st.info(f"📅 {len(upcoming)} upcoming reviews")
                for item in upcoming[:5]:  # Show next 5
                    review_date = datetime.fromisoformat(item['review_date'])
                    days_until = (review_date - today).days
                    st.write(f"• In {days_until} days: {item['question']['question'][:60]}...")
        else:
            st.info("Complete some practice sessions to build your review schedule!")
    
    with tab3:
        st.header("📊 Active Recall Analytics")
        
        completed = st.session_state.active_recall_data.get('completed_questions', [])
        
        if completed:
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Questions Completed", len(completed))
            
            with col2:
                recent_questions = completed[-10:] if len(completed) > 10 else completed
                hard_percentage = sum(1 for q in recent_questions if q.get('difficulty_rating') == 'hard') / len(recent_questions) * 100
                st.metric("Challenge Rate", f"{hard_percentage:.0f}%")
            
            with col3:
                if len(completed) > 1:
                    improvement = "Improving" if completed[-1].get('difficulty_rating') != 'hard' else "Challenging"
                    st.metric("Trend", improvement)
            
            # Difficulty distribution over time
            if len(completed) >= 5:
                dates = [datetime.fromisoformat(q['completed_at']).strftime("%m/%d") for q in completed[-10:]]
                difficulties = [q.get('difficulty_rating', 'medium') for q in completed[-10:]]
                
                # Create difficulty timeline
                difficulty_scores = {'easy': 1, 'medium': 2, 'hard': 3}
                scores = [difficulty_scores.get(d, 2) for d in difficulties]
                
                fig = px.line(
                    x=dates,
                    y=scores,
                    title="📈 Learning Difficulty Over Time",
                    labels={'x': 'Date', 'y': 'Difficulty Level'}
                )
                fig.update_traces(line_color='#764ba2', line_width=3)
                fig.update_layout(
                    yaxis=dict(tickvals=[1, 2, 3], ticktext=['Easy', 'Medium', 'Hard']),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start practicing to see your active recall analytics!")

def collaboration_hub():
    """Collaborative study sessions and group features"""
    st.markdown("### 👥 Collaborative Learning Hub")
    
    # Initialize collaboration data
    if 'collaboration_data' not in st.session_state:
        st.session_state.collaboration_data = {
            'study_groups': [],
            'shared_documents': [],
            'group_discussions': [],
            'peer_reviews': []
        }
    
    tab1, tab2, tab3 = st.tabs(["👥 Study Groups", "📚 Shared Documents", "💬 Discussions"])
    
    with tab1:
        st.header("🤝 Study Groups")
        
        # Create new study group
        with st.expander("➕ Create New Study Group"):
            group_name = st.text_input("Group Name", placeholder="e.g., Quantum Physics Study Group")
            group_description = st.text_area("Description", placeholder="What will this group focus on?")
            max_members = st.number_input("Max Members", min_value=2, max_value=20, value=5)
            
            if st.button("Create Group"):
                new_group = {
                    'id': len(st.session_state.collaboration_data['study_groups']),
                    'name': group_name,
                    'description': group_description,
                    'max_members': max_members,
                    'members': ['You'],
                    'created_at': datetime.now().isoformat(),
                    'activity_score': 0
                }
                st.session_state.collaboration_data['study_groups'].append(new_group)
                st.success(f"✅ Created study group: {group_name}")
                st.rerun()
        
        # Display existing groups (demo data)
        st.subheader("🌟 Available Study Groups")
        
        demo_groups = [
            {
                'name': 'Advanced Physics Concepts',
                'members': 8,
                'max_members': 12,
                'description': 'Exploring quantum mechanics and relativity',
                'activity': 'High',
                'focus_topics': ['quantum mechanics', 'relativity', 'thermodynamics']
            },
            {
                'name': 'Biology Study Circle',
                'members': 5,
                'max_members': 8,
                'description': 'Cell biology and molecular processes',
                'activity': 'Medium',
                'focus_topics': ['cell biology', 'genetics', 'evolution']
            },
            {
                'name': 'Mathematics Problem Solvers',
                'members': 10,
                'max_members': 15,
                'description': 'Calculus and advanced mathematics',
                'activity': 'High',
                'focus_topics': ['calculus', 'linear algebra', 'statistics']
            }
        ]
        
        for group in demo_groups:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{group['name']}**")
                    st.caption(group['description'])
                    st.write(f"Topics: {', '.join(group['focus_topics'])}")
                
                with col2:
                    st.metric("Members", f"{group['members']}/{group['max_members']}")
                    st.write(f"Activity: {group['activity']}")
                
                with col3:
                    if st.button(f"Join Group", key=f"join_{group['name']}"):
                        st.success(f"Joined {group['name']}!")
                    if st.button(f"View Details", key=f"view_{group['name']}"):
                        st.info("Group details would open here")
                
                st.markdown("---")
    
    with tab2:
        st.header("📚 Shared Study Materials")
        
        if st.session_state.current_pdf_data:
            st.subheader("📤 Share Current Document")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"Current Document: {st.session_state.current_pdf_data['name']}")
                share_note = st.text_area(
                    "Add a note for the group:",
                    placeholder="Why are you sharing this? What should the group focus on?"
                )
            
            with col2:
                share_with = st.selectbox(
                    "Share with Group:",
                    ["Advanced Physics Concepts", "Biology Study Circle", "Mathematics Problem Solvers"]
                )
                
                if st.button("📤 Share Document"):
                    shared_doc = {
                        'document': st.session_state.current_pdf_data['name'],
                        'shared_by': 'You',
                        'group': share_with,
                        'note': share_note,
                        'shared_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'downloads': 0,
                        'comments': []
                    }
                    
                    st.session_state.collaboration_data['shared_documents'].append(shared_doc)
                    st.success(f"✅ Shared with {share_with}")
        
        # Display shared documents
        st.subheader("📋 Recently Shared Documents")
        
        # Demo shared documents
        demo_shared_docs = [
            {
                'document': 'Quantum_Mechanics_Advanced.pdf',
                'shared_by': 'Alice Chen',
                'group': 'Advanced Physics Concepts',
                'note': 'Great resource for understanding wave functions!',
                'shared_at': '2024-08-10 14:30',
                'downloads': 15,
                'rating': 4.8
            },
            {
                'document': 'Cell_Biology_Fundamentals.pdf',
                'shared_by': 'Bob Miller',
                'group': 'Biology Study Circle',
                'note': 'Perfect for exam prep - covers all major topics',
                'shared_at': '2024-08-09 09:15',
                'downloads': 23,
                'rating': 4.6
            }
        ]
        
        for doc in demo_shared_docs:
            with st.expander(f"📄 {doc['document']} (⭐ {doc['rating']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Shared by:** {doc['shared_by']}")
                    st.write(f"**Group:** {doc['group']}")
                    st.write(f"**Note:** {doc['note']}")
                
                with col2:
                    st.metric("Downloads", doc['downloads'])
                    st.write(f"Shared: {doc['shared_at']}")
                    
                    col1_btn, col2_btn = st.columns(2)
                    with col1_btn:
                        if st.button("📥 Download", key=f"download_{doc['document']}"):
                            st.success("Downloaded!")
                    with col2_btn:
                        if st.button("💬 Comment", key=f"comment_{doc['document']}"):
                            st.info("Comment feature would open here")
    
    with tab3:
        st.header("💬 Group Discussions")
        
        # Create new discussion
        with st.expander("➕ Start New Discussion"):
            discussion_title = st.text_input("Discussion Title")
            discussion_content = st.text_area("Your question or topic")
            discussion_group = st.selectbox(
                "Post to Group:",
                ["Advanced Physics Concepts", "Biology Study Circle", "Mathematics Problem Solvers"]
            )
            
            if st.button("Post Discussion"):
                st.success("Discussion posted!")
        
        # Display recent discussions
        st.subheader("🔥 Recent Discussions")
        
        demo_discussions = [
            {
                'title': 'Help with Schrödinger Equation interpretation',
                'author': 'Emma Watson',
                'group': 'Advanced Physics Concepts',
                'preview': 'I understand the mathematical formulation, but struggling with the physical interpretation...',
                'replies': 7,
                'last_activity': '2 hours ago',
                'tags': ['quantum mechanics', 'equations', 'interpretation']
            },
            {
                'title': 'Study group for upcoming exam',
                'author': 'David Lee',
                'group': 'Biology Study Circle',
                'preview': 'Anyone interested in forming a focused study group for the molecular biology exam?',
                'replies': 12,
                'last_activity': '4 hours ago',
                'tags': ['exam prep', 'study group', 'molecular biology']
            },
            {
                'title': 'Best resources for integration techniques?',
                'author': 'Sarah Johnson',
                'group': 'Mathematics Problem Solvers',
                'preview': 'Looking for recommendations on mastering integration by parts and substitution...',
                'replies': 5,
                'last_activity': '1 day ago',
                'tags': ['calculus', 'integration', 'resources']
            }
        ]
        
        for discussion in demo_discussions:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{discussion['title']}**")
                    st.caption(f"by {discussion['author']} in {discussion['group']}")
                    st.write(discussion['preview'])
                    
                    # Tags
                    tag_html = " ".join([f'<span style="background-color: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{tag}</span>' for tag in discussion['tags']])
                    st.markdown(tag_html, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Replies", discussion['replies'])
                    st.write(f"Last: {discussion['last_activity']}")
                    
                    if st.button("💬 Join Discussion", key=f"join_disc_{discussion['title']}"):
                        st.info("Discussion would open here")
                
                st.markdown("---")

def display_exam_analytics():
    """Display comprehensive exam analytics"""
    if st.session_state.exam_history:
        st.header("📊 Exam Performance Analytics")
        
        # Recent performance trend
        recent_exams = st.session_state.exam_history[-5:]
        if len(recent_exams) > 1:
            scores = [exam['score'] for exam in recent_exams]
            exam_numbers = list(range(len(recent_exams)))
            
            fig = px.line(
                x=exam_numbers,
                y=scores,
                title="Recent Performance Trend",
                labels={'x': 'Exam Number', 'y': 'Score (%)'}
            )
            fig.update_traces(line_color='#667eea', line_width=3, marker_size=8)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.subheader("📈 Performance Summary")
        
        exam_data = []
        for i, exam in enumerate(st.session_state.exam_history[-10:]):  # Last 10 exams
            exam_data.append({
                'Exam': f"#{i+1}",
                'Date': exam['date'][:10],
                'Score': f"{exam['score']:.1f}%",
                'Grade': exam['grade'],
                'Difficulty': exam['difficulty'].title(),
                'Questions': f"{exam['correct']}/{exam['total']}"
            })
        
        st.dataframe(exam_data, use_container_width=True)

if __name__ == "__main__":
    main()