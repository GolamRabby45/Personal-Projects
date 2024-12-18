import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

if __name__ == "__main__":
    # Load resampled training data and validation data
    X_train = pd.read_pickle("../data/processed/X_train_resampled.pkl")
    y_train = np.load("../data/processed/y_train_resampled.npy")
    X_val = pd.read_pickle("../data/processed/X_val.pkl")
    y_val = np.load("../data/processed/y_val.npy")

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    logreg.fit(X_train, y_train)

    y_val_pred_logreg = logreg.predict(X_val)
    y_val_pred_prob_logreg = logreg.predict_proba(X_val)[:,1]

    print("Logistic Regression Validation Metrics:")
    print(classification_report(y_val, y_val_pred_logreg))
    print("ROC-AUC:", roc_auc_score(y_val, y_val_pred_prob_logreg))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    y_val_pred_rf = rf.predict(X_val)
    y_val_pred_prob_rf = rf.predict_proba(X_val)[:,1]

    print("\nRandom Forest Validation Metrics:")
    print(classification_report(y_val, y_val_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_val, y_val_pred_prob_rf))

    # Save models
    os.makedirs("../outputs", exist_ok=True)
    joblib.dump(logreg, "../outputs/logreg_baseline.pkl")
    joblib.dump(rf, "../outputs/rf_baseline.pkl")

    print("Baseline models trained and saved.")
