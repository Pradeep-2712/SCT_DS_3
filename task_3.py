import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("/Users/pradeepgorai/Documents/SKILLCRAFT/task_3/SCT_DS_3/bank/bank.csv", sep=';')

# -------------------------------
# 🔹 Data Preprocessing (FIXED)
# -------------------------------

# Convert categorical columns using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# 🔹 Features & Target
# -------------------------------

X = df.drop('y_yes', axis=1)   # target column after encoding
y = df['y_yes']

# -------------------------------
# 🔹 Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🔹 Train Model (Basic)
# -------------------------------

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 🔹 Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# 🔹 Evaluation
# -------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# 🔹 Visualization
# -------------------------------

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# -------------------------------
# 🔹 Improved Model (Tuned)
# -------------------------------

model_tuned = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model_tuned.fit(X_train, y_train)

y_pred_tuned = model_tuned.predict(X_test)

print("\nTuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))