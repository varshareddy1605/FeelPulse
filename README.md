# README for Sentiment Analysis and Text Summarization API

## Overview
This project is a **FastAPI-based application** for:
1. **Text Sentiment Analysis**: Detects whether a text is positive, negative, neutral, or sarcastic.
2. **URL Analysis**: Extracts content from a webpage, summarizes it, and provides sentiment and sarcasm analysis.

It leverages machine learning models for sentiment classification, embeddings, and summarization, incorporating libraries like Scikit-learn, Sentence Transformers, and Hugging Face Transformers.

---

## Features
- **Sentiment Analysis**: Uses a trained SVM model on text inputs to classify sentiment.
- **Sarcasm Detection**: Simple rule-based sarcasm detection for common patterns.
- **Content Summarization**: Summarizes large text content using BART (or equivalent) transformer models.
- **URL Analysis**: Extracts and analyzes text from a given URL.
- **Embeddings with FAISS**: Efficiently retrieves relevant text chunks using semantic similarity.

---

## Requirements
### Libraries
The project relies on the following Python libraries:
- **Core**:
  - `fastapi`, `uvicorn`, `pydantic`
  - `numpy`, `pandas`
  - `nltk`, `sentence-transformers`
  - `transformers`, `faiss`, `scikit-learn`
  - `newspaper3k`, `pickle`, `os`
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Pre-trained Models
- **TF-IDF Vectorizer + SVM Classifier**:
  - Saved in the `./models/` directory.
- **Sentence Transformer**:
  - Downloads or uses `"all-MiniLM-L6-v2"` from Hugging Face.
- **Summarization Model**:
  - Uses `facebook/bart-large-cnn` from Hugging Face.

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Necessary Resources**:
   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("punkt")
   ```

---

## Usage
### Training the Sentiment Classifier
To train the SVM classifier on a custom dataset:
1. Place your dataset in the `./` directory as `analysis_results.csv`.
2. The dataset should include:
   - `cleaned_text`: Preprocessed text.
   - `label`: Sentiment labels (`1` = Positive, `-1` = Negative, `0` = Neutral).
3. Run the script to train and save the model:
   ```bash
   python main.py
   ```

### Running the API
Start the server locally:
```bash
uvicorn main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`.

### API Endpoints
1. **Analyze Text Sentiment**:
   - **Endpoint**: `/analyze_text`
   - **Method**: POST
   - **Request Body**:
     ```json
     {
       "text": "This is an example text."
     }
     ```
   - **Response**:
     ```json
     {
       "sentiment": "Positive"
     }
     ```

2. **Analyze URL**:
   - **Endpoint**: `/analyze_url`
   - **Method**: POST
   - **Request Body**:
     ```json
     {
       "url": "https://example.com"
     }
     ```
   - **Response**:
     ```json
     {
       "summary": "Short summary of the webpage content.",
       "sentiment": "Neutral"
     }
     ```

---

## Directory Structure
```
/models/             # Pre-trained vectorizer and classifier
analysis_results.csv # Dataset file (if retraining)
main.py              # Main FastAPI application
requirements.txt     # Python dependencies
README.md            # Project documentation
```

---

## Future Enhancements
- Improve sarcasm detection using machine learning.
- Add support for multilingual sentiment analysis.
- Provide Dockerfile for containerized deployment.

---

## Contributors
- K.Varsha Reddy
- Abhishek Boga

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
