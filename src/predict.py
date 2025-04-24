# src/predict.py

import pickle
import os
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean input text (same as in preprocess.py)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load model and vectorizer
model_path = 'model/best_model.pkl'
vectorizer_path = 'model/tfidf_vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Predict function
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "REAL News 📰" if prediction == 1 else "FAKE News 🚨"

# Example usage
if __name__ == "__main__":
    sample = input("Enter news text to classify:\n")
    result = predict_news(sample)
    print("\nPrediction:", result)
