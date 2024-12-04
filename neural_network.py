import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK Verilerini indirme
nltk.download('stopwords')
nltk.download('punkt')

# Veri Yükleme ve Temizleme
tweets = []
sentiments = []

with open('test_62k.txt', 'r', encoding='utf-8') as file:
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

# Veri Önişleme
stop_words = set(stopwords.words('english'))
tweets = [re.sub(r'[^a-zA-Z\s]', '', tweet.lower()) for tweet in tweets]
tweets = [' '.join([word for word in tweet.split() if word not in stop_words]) for tweet in tweets]

# TF-IDF Vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(tweets).toarray()
y = np.array(sentiments)

# PyTorch Tensörlere Dönüştürme
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Model Tanımlama
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Model Parametreleri
input_dim = X.shape[1]
hidden_dim = 128
output_dim = 3

# Eğitim Fonksiyonu
def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

# Değerlendirme Fonksiyonu
def evaluate_model(model, X_test, y_test):
    model.eval()
    outputs = model(X_test).detach().numpy()
    preds = np.argmax(outputs, axis=1)
    acc = accuracy_score(y_test, preds)
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.show()
    return acc

# Farklı Test Yöntemleri

# 1. Eğitim Setini Aynı Anda Test Verisi Olarak Kullanma
model = SentimentClassifier(input_dim, hidden_dim, output_dim)
model = train_model(model, X_tensor, y_tensor)
acc = evaluate_model(model, X_tensor, y_tensor)
print(f"Eğitim Setini Aynı Anda Test Verisi Olarak Kullanma - Doğruluk: {acc:.4f}")

# 2. 5-Fold Cross Validation
kf = KFold(n_splits=5)
fold_accuracies = []
for train_idx, test_idx in kf.split(X_tensor):
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
    
    model = SentimentClassifier(input_dim, hidden_dim, output_dim)
    model = train_model(model, X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    fold_accuracies.append(acc)

print(f"5-Fold Cross Validation Ortalama Doğruluk: {np.mean(fold_accuracies):.4f}")

# 3. 10-Fold Cross Validation
kf = KFold(n_splits=10)
fold_accuracies = []
for train_idx, test_idx in kf.split(X_tensor):
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
    
    model = SentimentClassifier(input_dim, hidden_dim, output_dim)
    model = train_model(model, X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    fold_accuracies.append(acc)

print(f"10-Fold Cross Validation Ortalama Doğruluk: {np.mean(fold_accuracies):.4f}")

# 4. %66-%34 Eğitim-Test Ayırma (5 Farklı Rassal Ayırma)
split_accuracies = []
for i in range(5):
    X_shuffled, y_shuffled = shuffle(X, y, random_state=i)
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.34, random_state=42)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    model = SentimentClassifier(input_dim, hidden_dim, output_dim)
    model = train_model(model, X_train_tensor, y_train_tensor)
    acc = evaluate_model(model, X_test_tensor, y_test_tensor)
    split_accuracies.append(acc)

print(f"%66-%34 Eğitim-Test Ayırma Ortalama Doğruluk: {np.mean(split_accuracies):.4f}")
