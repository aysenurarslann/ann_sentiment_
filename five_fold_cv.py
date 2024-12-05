import numpy as np
from sklearn.model_selection import KFold
import torch
from training import train_model, evaluate_model
from model import DeepSentimentClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class MediumSentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(MediumSentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def five_fold_cv(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate):
    print("\n--- 5-Fold Cross Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in kf.split(X_tensor):
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
        
        model = MediumSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
        model = train_model(model, X_train, y_train, epochs=15, lr=0.0005)
        acc = evaluate_model(model, X_test, y_test)
        fold_accuracies.append(acc)

    print(f"5-Fold Cross Validation Ortalama Doğruluk: {np.mean(fold_accuracies):.4f}")
