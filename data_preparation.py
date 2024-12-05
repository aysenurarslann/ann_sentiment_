import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords

# NLTK Verilerini İndirme
nltk.download('stopwords')
nltk.download('punkt')

# Veri Yükleme
def load_data(file_path):
    tweets, sentiments = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            try:
                sentiment = int(parts[0])
                tweet = parts[1]
                tweets.append(tweet)
                sentiments.append(sentiment)
            except ValueError:
                continue
    return tweets, sentiments

# Veri Önişleme
def preprocess_tweets(tweets):
    stop_words = set(stopwords.words('english'))
    important_words = {"not", "değil"}  # Önemli bağlamsal kelimeler
    stop_words = stop_words - important_words
    tweets = [re.sub(r'http\S+|www\S+|https\S+', '', tweet) for tweet in tweets]  # URL'leri kaldır
    tweets = [re.sub(r'[^a-zA-Z\s]', '', tweet.lower()) for tweet in tweets]
    tweets = [' '.join([word for word in nltk.word_tokenize(tweet) if word not in stop_words]) for tweet in tweets]
    return tweets

# Özellik Çıkarımı (TF-IDF)
def extract_features(tweets, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X = tfidf_vectorizer.fit_transform(tweets).toarray()
    return X, tfidf_vectorizer

# SMOTE ile Veri Dengesizliğini Giderme
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced
