import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("vibration_features_dataset.csv")

# Features
X = df.drop("label", axis=1)

# Labels
y = df["label"]

# ---------------------------------
# Split Dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# Train Model
# ---------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------
# Predictions
# ---------------------------------

predictions = model.predict(X_test)

# ---------------------------------
# Results
# ---------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("Predictive Maintenance AI Model")
print("==============================\n")

print(f"Model Accuracy: {accuracy*100:.2f}%\n")

print("Confusion Matrix\n")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report\n")
print(classification_report(y_test, predictions))