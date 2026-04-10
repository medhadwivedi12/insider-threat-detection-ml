import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and labels
X = df[['login_attempts', 'file_access_count', 'unusual_activity']]
y = df['label'].map({'normal': 0, 'threat': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ML Model (Classification)
ml_model = DecisionTreeClassifier()
ml_model.fit(X_train, y_train)
ml_pred = ml_model.predict(X_test)

# Anomaly Detection Model
anomaly_model = IsolationForest(contamination=0.4)
anomaly_model.fit(X_train)

anomaly_pred = anomaly_model.predict(X_test)
anomaly_pred = [1 if x == -1 else 0 for x in anomaly_pred]

# Hybrid Decision (Fusion)
hybrid_pred = []
for i in range(len(ml_pred)):
    if ml_pred[i] == 1 or anomaly_pred[i] == 1:
        hybrid_pred.append(1)
    else:
        hybrid_pred.append(0)

# Evaluation Metrics
accuracy = accuracy_score(y_test, hybrid_pred)
precision = precision_score(y_test, hybrid_pred)
recall = recall_score(y_test, hybrid_pred)
f1 = f1_score(y_test, hybrid_pred)

print("\n--- HYBRID MODEL RESULTS ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Graph
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values)
plt.title("Hybrid Model Performance")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.show()
