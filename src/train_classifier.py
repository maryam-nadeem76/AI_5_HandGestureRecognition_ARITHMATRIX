"""
Step 2: Train the Classifier
------------------------------
Trains a simple, fast classifier on the hand landmark data collected in
collect_data.py. We use a Random Forest here rather than a deep neural
network deliberately -- with only a few hundred samples and just 63
input features (21 landmarks x 3 coordinates), a small classical model
is the right-sized tool for this data volume, the same lesson from your
BCI project about matching model complexity to data size.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_PATH = "../data/gestures.csv"
MODEL_PATH = "../models/gesture_classifier.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} gestures")
    print(df['label'].value_counts())

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues')
    plt.title("Hand Gesture Classifier — Confusion Matrix")
    plt.savefig("../confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Confusion matrix saved to confusion_matrix.png")
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()