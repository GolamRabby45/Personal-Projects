import sys
import pandas as pd
import numpy as np
import joblib
import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <input_csv>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)

    # Load the final model (rf_tuned for demonstration)
    model = joblib.load("../outputs/rf_tuned.pkl")

    # We need to replicate the preprocessing done before:
    # Load scaler if used separately. In previous steps we used inline scaling.
    # For simplicity, we assume the input data is already preprocessed or matches the original schema.
    # In a real scenario, you would reload and apply the same scaler used in preprocess.py.
    
    # For demonstration, let's assume the input CSV contains columns: ['Time', 'V1', 'V2', ..., 'V28', 'Amount', 'Hour'] exactly as during training.
    data = pd.read_csv(input_file)
    # Ensure columns order:
    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else data.columns
    data = data[expected_columns]

    # Predict probabilities
    y_prob = model.predict_proba(data)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)  # Simple threshold at 0.5, adjust if needed

    # Print results
    results = pd.DataFrame({
        "Fraud_Probability": y_prob,
        "Predicted_Class": y_pred
    })
    print(results)
