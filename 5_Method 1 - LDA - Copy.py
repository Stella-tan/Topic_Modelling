import streamlit as st
import PyPDF2  # extract text
import re  # text cleaning: removing punctuation and newlines
import spacy  # remove stopwords & lemma
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from transformers import BartTokenizer, BartForConditionalGeneration

st.title("LDA Topic Modelling")

# Custom CSS to style the Streamlit app
st.markdown("""
<style>
/* Sidebar Styling */
[data-testid="stSidebar"] {
  background-color: black; 
  color: #white;
}

/* Mainpage - Dashboard */

[data-testid="stAppViewContainer"] {
  background-image: url("https://s1.aigei.com/src/img/gif/bc/bca9742645ca4210b7df6c3353e75f19.gif?imageMogr2/auto-orient/thumbnail/!282x282r/gravity/Center/crop/282x282/quality/85/%7CimageView2/2/w/282&e=1735488000&token=P7S2Xpzfz11vAkASLTkfHN7Fw-oOZBecqeJaxypL:SDqdOE3d_qtENhN4QrmniY-YMzU=");
  background-size: cover; 
  background-position: center;
  background-repeat: no-repeat; 
}
[data-testid="stApp"], [data-testid="stHeader"] {
  background-color: black;
  color: white;
}
/* Header Styling */
.stApp h1 {
  text-align: center;
  font-size: 4em;
  color: #bbc7ff;
}
.stApp h3 {
  font-size: 1.5em;
  color: #bbc7ff;  
}
      
p {
  color: white;            
}
/*Buttons*/
[data-testid="stFileUploaderDropzone"] {
  background-color: #3f4a6f;
}
[data-testid="stBaseButton-secondary"] {
  color: white;
  background-color: transparent;
}
[data-testid="stBaseButton-secondary"]:hover {
  background-color: white;
  color: #3f4a6f;
  border-color: white;
}
            
.stButton button {
  background-color: transparent;
  border-color: white;
}
.stButton button:hover {
  border-color: white;
  box-shadow: 0 5px #666;
  transform: translateY(4px);
  color: #3f4a6f;
}
.stButton button p{
  color: white;
}
.stButton button p:hover {
  color: #3f4a6f;
}
.st-emotion-cache-1n47svx:focus:not(:active){
  border-color: white;
  color: white;
}

/*Slider css*/
.st-bq, .st-bb, .st-be, .st-bx, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, .st-bw, .st-hx, .st-in, .st-ik, .st-il, .st-im, .st-ix, .st-iy, .st-iz, .st-j0{
  background-image: linear-gradient(to right, white, #670067, white, darkblue);
}        
[data-testid="stSliderThumbValue"] {
  color: white ;
}
.stProgress>div {
  background-color: white !important;
}
.st-emotion-cache-1dj3ksd {
  background-color: white;
}

/*Result or Output css*/
.topic p {
  color: white;
  font-size: 1.2em;
  

}

.warning-text {
    color: #ff4b4b; 
    font-size: 18px; 
    font-weight: bold; 
    background-color: #ffe6e6; 
    padding: 10px; 
    border-radius: 5px; 
}

/* Footer Styling */
footer {
  text-align: center;
  margin-top: 40px;
  font-size: 14px;
  color: #777;
}
</style>
""", unsafe_allow_html=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)

# Helper functions
def extract_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    """Clean text by removing newlines and extra spaces."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def remove_numbers(text):
    """Remove numbers from the text."""
    return re.sub(r'\b\d+\b', '', text)

def get_top_words_for_topic(model, vectorizer, num_words=10):
    """Get top words for each topic."""
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    return topics

# Function to assign topic labels
def assign_topic_labels(model, vectorizer, num_words=3):
    feature_names = vectorizer.get_feature_names_out()
    used_top_words = set()  # Track used top words to avoid repetition
    
    for topic_idx, topic in enumerate(model.components_):
        # Get the indices of the top 'num_words' words
        top_words_idx = np.argsort(topic)[::-1][:num_words]
        
        # Get the corresponding words
        top_words = [feature_names[i] for i in top_words_idx]
        
        # Check if the top word is repeated from previous topics
        if top_words[0] in used_top_words:
            # If repeated, try displaying the second most important word
            if top_words[1] not in used_top_words:
                top_words[0] = top_words[1]
            else:
                # If both the first and second words are repeated, display the third word
                top_words[0] = top_words[2]
        
        # Add the used top word to the set (only the word that's displayed)
        used_top_words.add(top_words[0])
        
        # Return HTML with CSS class for styling
        yield f'<div class="topic"><p><strong>Topic {topic_idx+1}:</strong> {top_words[0]}</p></div>'

def calculate_coherence_score(lemmatized_tokens_list, topics, coherence_measure='c_v'):
    """Calculate coherence score for topics."""
    dictionary = Dictionary(lemmatized_tokens_list)
    coherence_model = CoherenceModel(
        topics=topics,
        texts=lemmatized_tokens_list,
        dictionary=dictionary,
        coherence=coherence_measure
    )
    return coherence_model.get_coherence()

def summarize_text_with_bart(text):
    inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True, padding=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,  # You can adjust max_length for a longer or shorter summary
        min_length=20,
        num_beams=4,
        length_penalty=0.8,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI Design --------------------------------------------------------------------------------------------------------------------------------------

# Title and Description
st.markdown("""<p style="text-align: center; font-size: 1.2em;">Upload your PDFs, analyze the content, and discover hidden topics.</p>""", unsafe_allow_html=True)

# File Uploader for PDFs
st.markdown("""
    <div>
        <h3><strong>Upload PDF Files</strong></h3>
        <p>Upload multiple PDF files to analyze their content.</p>
    </div>
""", unsafe_allow_html=True)

placeholder = st.empty()

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    placeholder = st.empty()
    placeholder.info("Uploaded successfully!")
    time.sleep(5)
    placeholder.empty()
    # st.balloons()

st.markdown("""<div><h3>⚙️ Select Number of Topics</h3></div>""", unsafe_allow_html=True)

# Topic Settings (Placed after file upload section)
num_topics = st.slider("", min_value=2, max_value=10, value=3)

# Progress Bar
progress_bar = st.progress(0)

# Process Button Logic
process_button = st.button("Process PDFs")

checking = False

if uploaded_files: 
    checking = True
else: 
    checkign = False

if process_button and uploaded_files:
    st.write("Processing... Please wait.") 

    documents_to_process = []
    for pdf_file in uploaded_files:  # Correct this line
        progress_bar.progress(int((uploaded_files.index(pdf_file) + 1) / len(uploaded_files) * 100))

        text = extract_pdf(pdf_file)
        text = remove_numbers(text)
        cleaned_text = clean_text(text).lower()
        doc = nlp(cleaned_text)
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        documents_to_process.append(lemmatized_tokens)

    # Prepare documents for TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
    tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in documents_to_process])

    # Apply LDA for topic modeling
    lda_model = LatentDirichletAllocation(
        n_components=num_topics, 
        random_state=1, 
        learning_method='online', 
        max_iter=10, 
        learning_decay=0.7, 
        batch_size=128, 
        evaluate_every=-1, 
        n_jobs=-1 
    )
    lda_model.fit(tfidf_matrix)

    # Topic Identified Section Container
    st.header("Topics Identified")
    for topic_label in assign_topic_labels(lda_model, vectorizer):
        st.markdown(topic_label, unsafe_allow_html=True)

    # Topic Summary Section Container
    st.header("💬 Topic Summary")
    top_words_per_topic = get_top_words_for_topic(lda_model, vectorizer)
    for topic_idx, top_words in enumerate(top_words_per_topic):
        # Join the top words into a summary text and pass to BART
        summary_text = ' '.join(top_words)
        summary = summarize_text_with_bart(summary_text)
        st.markdown(f'<div class="topic"><p><strong>Topic {topic_idx + 1}:</strong> {summary}</p></div>', unsafe_allow_html=True)

    # Calculate coherence score
    coherence_score = calculate_coherence_score(documents_to_process, top_words_per_topic)
    print("Coherence Score:", coherence_score)

    # Reset progress bar to 100% once done
    progress_bar.progress(100)
elif process_button and checking is False:
    st.markdown('<div class="warning-text">Please upload at least one PDF!</div>', unsafe_allow_html=True)
else:
    st.write("Upload PDFs and click 'Process' to get started!")  

# Footer
st.markdown("""
    <footer>
        <p>Created with 🤍 by TRP & William Team</p>
    </footer>
""", unsafe_allow_html=True)
