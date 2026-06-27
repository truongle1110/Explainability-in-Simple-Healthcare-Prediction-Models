import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
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

# Pipeline: scale first, then train MLP
nn_model = Pipeline(
    steps=[
        (
            'mlp',
            MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                alpha=1e-5,
                learning_rate_init=1e-5,
                max_iter=10000,
                early_stopping=False,
                n_iter_no_change=500,
                verbose=True,
                random_state=42,
            ),
        ),
    ]
)

nn_model.fit(X_train, y_train)

y_pred = nn_model.predict(X_test)
y_proba = nn_model.predict_proba(X_test)[:, 1]

print('=== Neural Network (MLPClassifier) ===')
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

# Plot training loss per iteration
mlp = nn_model.named_steps['mlp']
loss_values = mlp.loss_curve_

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', markersize=3)
plt.title('MLP Training Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Training stopped after {mlp.n_iter_} iterations')
