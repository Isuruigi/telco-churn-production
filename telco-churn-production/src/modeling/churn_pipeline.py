import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChurnPipeline:
    def __init__(self, random_state=42, **kwargs):
        self.model = LogisticRegression(random_state=random_state, **kwargs)

    def train(self, X_train, y_train):
        """Trains the Logistic Regression model."""
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def predict(self, X_test):
        """Makes predictions using the trained model."""
        print("Making predictions...")
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """Evaluates the model's performance."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"--- End Model Evaluation ---")
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

    def save_model(self, model_path):
        """Saves the trained model to the specified path."""
        try:
            joblib.dump(self.model, model_path)
            print(f"Model successfully saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, model_path):
        """Loads a trained model from the specified path."""
        try:
            self.model = joblib.load(model_path)
            print(f"Model successfully loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
