# src/train_model.py

from preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_and_save_model(data_path, model_path='model/best_model.pkl'):
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Step 2: Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 3: Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Step 4: Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to {model_path}")

# Run if this file is executed directly
if __name__ == "__main__":
    train_and_save_model("data/fake_news.csv")
