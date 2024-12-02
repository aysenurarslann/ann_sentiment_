import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Veri indirmek için
nltk.download('stopwords')
nltk.download('punkt')

# Veri yükleme ve işleme
tweets = []
sentiments = []

# Veriyi yükleme ve temizleme
with open('test_62k.txt', 'r',encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')  # Tab ile ayır
        if len(parts) != 2:
            print(f"Yanlış formatlı satır: {line}")
            continue
        try:
            sentiment = int(parts[0])  # Etiketi tam sayıya çevir
            tweet = parts[1]
            tweets.append(tweet)
            sentiments.append(sentiment)
        except ValueError:
            print(f"Etiket hatası: {line}")
            continue

# 2. Veri Temizliği
tweets = [tweet.lower() for tweet in tweets]  # Küçük harfe çevir
tweets = [re.sub(r'[^a-zA-Z\s]', '', tweet) for tweet in tweets]  # Özel karakterleri kaldır
stop_words = set(stopwords.words('english'))
tweets = [' '.join([word for word in tweet.split() if word not in stop_words]) for tweet in tweets]

# 3. Tokenizasyon
tweets = [word_tokenize(tweet) for tweet in tweets]

# 4. TF-IDF ile vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform([' '.join(tweet) for tweet in tweets]).toarray()
y = np.array(sentiments)

# 5. Veri setini eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

# 6. Veriyi PyTorch tensörlerine dönüştürme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 7. Veri yükleyiciler
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 8. PyTorch ile model tanımlama
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

# Model parametreleri
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = 3  # 3 sınıf (pozitif, negatif, nötr)

# Model oluşturma
model = SentimentClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. Model eğitimi
epochs = 20
train_losses, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    # Doğrulama setinde değerlendirme
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# 10. Performans görselleştirme
plt.plot(train_losses, label='Eğitim Kaybı')
plt.title('Eğitim Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

plt.plot(val_accuracies, label='Doğrulama Doğruluğu')
plt.title('Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# 11. Test seti üzerinde değerlendirme
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_accuracy = correct / total
print(f"Test Seti Doğruluğu: {test_accuracy:.4f}")
