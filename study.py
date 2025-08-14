import streamlit as st
import PyPDF2

st.set_page_config(page_title="StudyPDF", layout="wide")
st.title("📚 StudyPDF - Simple PDF Reader")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Create PDF reader
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    # Extract text from all pages
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Show extracted text
    st.subheader("Extracted Text:")
    st.write(text)

    # Optional: Download extracted text
    st.download_button(
        label="Download Text File",
        data=text,
        file_name="extracted_text.txt",
        mime="text/plain"
    )
else:
    st.info("Please upload a PDF to get started.")
    import streamlit as st
import PyPDF2
from transformers import pipeline

# Load the Hugging Face Q&A model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

st.set_page_config(page_title="StudyPDF Q&A", layout="wide")
st.title("📚 StudyPDF - Ask Questions from Your PDF")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"

    st.success("✅ PDF loaded successfully!")

    # Display text preview
    with st.expander("Preview Extracted Text"):
        st.write(pdf_text[:1500] + "..." if len(pdf_text) > 1500 else pdf_text)

    # Question input
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Searching for the answer..."):
            result = qa_pipeline(question=question, context=pdf_text)
        st.subheader("Answer:")
        st.write(result["answer"])
        st.caption(f"Confidence: {result['score']:.2%}")

else:
    st.info("Please upload a PDF to start.")