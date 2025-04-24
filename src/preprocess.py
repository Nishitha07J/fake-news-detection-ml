# src/preprocess.py

import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Clean the text
    df['clean_text'] = df['text'].apply(clean_text)

    # Features and labels
    X = df['clean_text']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Save the vectorizer
    with open('model/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    return X_train_vec, X_test_vec, y_train, y_test
