from sklearn.metrics import accuracy_score, classification_report
from genai.llm_explainer import generate_explanation
import numpy as np

def evaluate_model(model, X_test, y_test):
    predictions = (model.predict(X_test) > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


# Example sequence & prediction
sequence = "ATGCGTACGTAGCTAGCTAG"
prediction = 1  # output from model

explanation = generate_explanation(sequence, prediction)
print("GenAI Explanation:")
print(explanation)
