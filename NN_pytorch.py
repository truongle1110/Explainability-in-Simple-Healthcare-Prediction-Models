import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Load data
data = pd.read_csv('dataset/diabetes_scaled.csv')

target = 'Outcome'
X = data.drop(columns=[target])
y = data[target]

# Stratify to keep label distribution stable across train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


class DiabetesNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = DiabetesNN(input_dim=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

max_epochs = 1000

train_loss_values = []
test_loss_values = []

for epoch in range(1, max_epochs + 1):
    model.train()
    optimizer.zero_grad()
    train_logits = model(X_train_tensor)
    train_loss = criterion(train_logits, y_train_tensor)
    train_loss.backward()
    optimizer.step()

    train_loss_values.append(train_loss.item())

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_loss = criterion(test_logits, y_test_tensor)
    test_loss_values.append(test_loss.item())

    print(f'Epoch {epoch:4d} | train_loss={train_loss.item():.8f} | test_loss={test_loss.item():.8f}')

model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    y_proba = torch.sigmoid(test_logits).cpu().numpy().ravel()
    y_pred = (y_proba >= 0.5).astype(int)

print('=== Neural Network (PyTorch) ===')
print(classification_report(y_test, y_pred, digits=4))
print(f'ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}')

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

confusion_df = pd.DataFrame(
    cm,
    index=['Actual 0 (No Diabetes)', 'Actual 1 (Diabetes)'],
    columns=['Predicted 0', 'Predicted 1'],
)

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network - Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot train and test loss curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Train Loss', linewidth=1.6)
plt.plot(range(1, len(test_loss_values) + 1), test_loss_values, label='Test Loss', linewidth=1.6)
plt.title('MLP Train/Test Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f'Training finished after {max_epochs} epochs')
