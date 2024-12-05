import torch
from training import train_model, evaluate_model
from model import DeepSentimentClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class SimpleSentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(SimpleSentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_and_test_same(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate):
    print("\n--- Eğitim Setini Aynı Zamanda Test Verisi Olarak Kullanma ---")
    model = SimpleSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
    model = train_model(model, X_tensor, y_tensor, epochs=15, lr=0.0005)
    acc = evaluate_model(model, X_tensor, y_tensor)
    print(f"Eğitim Seti Doğruluğu: {acc:.4f}")
