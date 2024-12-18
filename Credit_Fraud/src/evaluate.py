import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import shap
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Load test data
    X_test = pd.read_pickle("../data/processed/X_test.pkl")
    y_test = np.load("../data/processed/y_test.npy")

    # Load the best model (for demonstration, we’ll use the stacking ensemble)
    model_path = "../outputs/stacking_meta_clf.pkl"
    if not os.path.exists(model_path):
        # Fall back to a tuned RF if ensemble not available
        model_path = "../outputs/rf_tuned.pkl"

    model = joblib.load(model_path)

    # To get predictions, we need to replicate what we did in ensemble step if using stacking_meta_clf:
    # If stacking was used, we need predictions from the base models on X_test.
    # Let's assume we are just evaluating the tuned RF or a single model for simplicity:
    # If you're evaluating stacking, you must produce meta features as in the tuning step.
    # For simplicity, let's evaluate the tuned RF model directly.
    if "stacking_meta_clf" in model_path:
        # If stacked, load base models and produce meta-features for X_test
        logreg = joblib.load("../outputs/logreg_baseline.pkl")
        rf_baseline = joblib.load("../outputs/rf_baseline.pkl")
        xgb = joblib.load("../outputs/xgb_model_final.pkl")

        test_pred_logreg = logreg.predict_proba(X_test)[:,1]
        test_pred_rf = rf_baseline.predict_proba(X_test)[:,1]
        test_pred_xgb = xgb.predict_proba(X_test)[:,1]
        X_test_meta = np.column_stack((test_pred_logreg, test_pred_rf, test_pred_xgb))

        y_test_pred = model.predict(X_test_meta)
        y_test_prob = model.predict_proba(X_test_meta)[:,1]
    else:
        # If using a single model (like rf_tuned)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:,1]

    print("Test Set Metrics:")
    print(classification_report(y_test, y_test_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC:", pr_auc)

    # SHAP Interpretation (only for models that support TreeExplainer, e.g., XGBoost or RF)
    # If using a tree-based model (rf_tuned or xgb), we can do:
    if "rf_tuned" in model_path:
        explainer = shap.TreeExplainer(model)
        # Take a sample of test data for speed
        X_test_sample = X_test.sample(n=100, random_state=42)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Plot SHAP summary
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values[1], X_test_sample, plot_type="bar", show=False)
        plt.title("Feature Importance via SHAP")
        plt.savefig("../outputs/shap_summary.png", bbox_inches='tight')
        plt.close()

        print("SHAP summary plot saved.")

    print("Evaluation complete.")
