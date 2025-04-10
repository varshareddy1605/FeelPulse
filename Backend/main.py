import re
import pickle
import os
import nltk
import faiss
import torch
import uvicorn
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from newspaper import Article
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -------------------------------
# Ensure necessary NLTK resources
# -------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

# -----------------------------
# 1) Training + Model Saving
# -----------------------------

def train_and_save_model():
    """
    Trains an SVM sentiment classifier on analysis_results.csv 
    and saves the TF-IDF vectorizer + classifier into ./models/.
    """

    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)

    file_path = "./analysis_results.csv"  # Adjust if your CSV is elsewhere
    if not os.path.exists(file_path):
        print(f"[WARNING] Dataset file '{file_path}' not found. Skipping training.")
        return

    df = pd.read_csv(file_path)

    # Drop missing values
    df = df.dropna(subset=["cleaned_text", "label"])

    # Convert label to integer
    df["label"] = df["label"].astype(int)

    # Preprocessing function
    def preprocess_text(text):
        st_words = set(stopwords.words("english"))
        tokens = nltk.word_tokenize(text.lower())  # Lowercase and tokenize
        tokens = [word for word in tokens if word.isalnum() and word not in st_words]
        return " ".join(tokens)

    # Apply preprocessing
    df["processed_text"] = df["cleaned_text"].apply(preprocess_text)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"], df["label"], test_size=0.2, random_state=42
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Hyperparameter tuning for LinearSVC
    param_grid = {"C": [0.1, 1, 10]}
    grid_search = GridSearchCV(
        LinearSVC(dual=False, class_weight="balanced"),
        param_grid,
        cv=3
    )
    grid_search.fit(X_train_tfidf, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate model
    y_pred = best_model.predict(X_test_tfidf)
    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    with open("./models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("./models/classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\nModel training and evaluation completed. Saved model and vectorizer.")

# -------------------------------------------------------------------
# 2) FastAPI Initialization + Loading All Pre-Trained / Saved Models
# -------------------------------------------------------------------

# OPTIONAL: Uncomment the line below if you want to re-train
# the model each time you start the server.
# train_and_save_model()

# Check if the models exist; if not, attempt training:
if not os.path.exists("./models/vectorizer.pkl") or not os.path.exists("./models/classifier.pkl"):
    train_and_save_model()

# Initialize FastAPI
app = FastAPI()

# Enable CORS (so that any frontend can call these endpoints)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained vectorizer and classifier
try:
    with open("./models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("./models/classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
except:
    vectorizer = None
    classifier = None
    print("[ERROR] Could not load vectorizer/classifier from './models/'.")

# Load Sentence Transformer for embeddings (for URL analysis)
# This presumes you've downloaded or placed 'all-MiniLM-L6-v2' into ./models/
# Or you can load it from huggingface:  SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
try:
    embedding_model = SentenceTransformer("./models/all-MiniLM-L6-v2")
except:
    # Fallback to HF Hub if local model not found:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Summarization Model + Tokenizer (Bart, T5, etc.)
# This presumes you've placed 'bart-large-cnn' into ./models/
# or you can load directly from HF Hub by passing "facebook/bart-large-cnn"
try:
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained("./models/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("./models/bart-large-cnn")
except:
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# -----------------------------------
# 3) Helper Functions (Text Analysis)
# -----------------------------------

def detect_sarcasm_rule_based(text: str) -> bool:
    """
    Simple rule-based sarcasm detection using keywords.
    Expand or modify patterns as needed.
    """
    sarcasm_keywords = [
        "yeah right",
        "oh great",
        "just what i needed",
        "oh really",
        "oh wow",
        "yeah sure",
        "as if",
        "totally helpful",
        "best day ever",
        "worst day ever"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in sarcasm_keywords)

def preprocess_text(text: str) -> str:
    """Regex-based text cleaning + lowercasing for sentiment analysis."""
    # Remove non-alphanumeric, separate words
    tokens = re.findall(r"\b\w+\b", text.lower())
    return " ".join(tokens)

def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment using the trained SVM model.  
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    if not vectorizer or not classifier:
        return "Neutral"  # fallback if no model is loaded
    processed = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed])
    pred = classifier.predict(text_tfidf)[0]
    # Our training used label=1 for Positive, -1 for Negative, 0 for Neutral (?)
    # Adjust if your dataset labeling is different
    if pred == 1:
        return "Positive"
    elif pred == -1:
        return "Negative"
    else:
        return "Neutral"

# ---------------------------------------------
# 4) Helper Functions (URL Summarization Flow)
# ---------------------------------------------

def extract_text_from_url(url: str) -> str:
    """
    Extracts text from a webpage using Newspaper3k.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def chunk_text(text: str, chunk_size: int = 500):
    """Split text into smaller chunks (roughly)."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(text_chunks):
    """Generate embeddings using SBERT and store in FAISS index."""
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, text_chunks, embeddings

def retrieve_information(query: str, index, text_chunks, top_k=3):
    """Retrieve relevant chunks using FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_text = " ".join([text_chunks[i] for i in indices[0]])
    return retrieved_text

def summarize_text(text: str) -> str:
    """Summarizes text using a Transformer model (e.g. BART)."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs.input_ids, max_length=200, min_length=50)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ----------------------------
# 5) Pydantic Request Schemas
# ----------------------------

class SentimentRequest(BaseModel):
    text: str

class URLInput(BaseModel):
    url: str

# -----------------------
# 6) FastAPI Endpoints
# -----------------------

@app.post("/analyze_text")
def analyze_text_sentiment(request: SentimentRequest):
    """
    Endpoint for analyzing direct text input for sentiment / sarcasm.
    """
    text = request.text
    # Sarcasm check
    if detect_sarcasm_rule_based(text):
        return {"sentiment": "Sarcastic"}
    # Otherwise, do normal sentiment
    sentiment = analyze_sentiment(text)
    return {"sentiment": sentiment}

@app.post("/analyze_url")
def analyze_url(input_data: URLInput):
    """
    Endpoint for summarizing a URL's content and analyzing sentiment + sarcasm.
    """
    try:
        # 1) Extract the raw text from URL
        text = extract_text_from_url(input_data.url)

        # 2) Chunk & embed text
        text_chunks = chunk_text(text)
        index, text_chunks, embeddings = generate_embeddings(text_chunks)

        # 3) Retrieve relevant chunks for "Summarize this text"
        retrieved_text = retrieve_information(
            "Summarize this text", index, text_chunks, top_k=3
        )

        # 4) Summarize the retrieved text
        summary = summarize_text(retrieved_text)

        # 5) Analyze sentiment of the summary
        sentiment = analyze_sentiment(summary)

        # 6) Check for sarcasm
        sarcasm_found = detect_sarcasm_rule_based(summary)
        final_sentiment = "Sarcastic" if sarcasm_found else sentiment

        return {
            "summary": summary,
            "sentiment": final_sentiment
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------
# 7) Main entry
# -------------

if __name__ == "__main__":
    # Run the server:  uvicorn main:app --reload
    # or simply:       python main.py
    uvicorn.run(app, host="127.0.0.1", port=8000)