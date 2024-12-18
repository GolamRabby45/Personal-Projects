import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

if __name__ == "__main__":
    # Load training and validation sets
    X_train = pd.read_pickle("../data/processed/X_train_resampled.pkl")
    y_train = np.load("../data/processed/y_train_resampled.npy")
    X_val = pd.read_pickle("../data/processed/X_val.pkl")
    y_val = np.load("../data/processed/y_val.npy")

    # Hyperparameter tuning for Random Forest
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"]
    }

    rand_search = RandomizedSearchCV(rf, param_dist, n_iter=10, scoring='roc_auc', 
                                     cv=3, random_state=42, n_jobs=-1, verbose=1)
    rand_search.fit(X_train, y_train)

    best_rf = rand_search.best_estimator_
    print("Best RF parameters:", rand_search.best_params_)
    y_val_pred_best_rf = best_rf.predict(X_val)
    y_val_prob_best_rf = best_rf.predict_proba(X_val)[:,1]
    print("Tuned RF Validation Metrics:")
    print(classification_report(y_val, y_val_pred_best_rf))
    print("ROC-AUC:", roc_auc_score(y_val, y_val_prob_best_rf))
    joblib.dump(best_rf, "../outputs/rf_tuned.pkl")

    # Ensemble (Stacking)
    # Load baseline models and XGBoost
    logreg = joblib.load("../outputs/logreg_baseline.pkl")
    rf_baseline = joblib.load("../outputs/rf_baseline.pkl")
    xgb = joblib.load("../outputs/xgb_model_final.pkl")

    # Create meta-features from the validation set
    val_pred_logreg = logreg.predict_proba(X_val)[:,1]
    val_pred_rf = rf_baseline.predict_proba(X_val)[:,1]
    val_pred_xgb = xgb.predict_proba(X_val)[:,1]

    X_val_meta = np.column_stack((val_pred_logreg, val_pred_rf, val_pred_xgb))

    # For final training of ensemble, we use the same approach on the training set
    # We need predictions on training set from each model (using cross-validation ideally)
    # For simplicity, we’ll just predict on X_train directly (not recommended for real project)
    train_pred_logreg = logreg.predict_proba(X_train)[:,1]
    train_pred_rf = rf_baseline.predict_proba(X_train)[:,1]
    train_pred_xgb = xgb.predict_proba(X_train)[:,1]
    X_train_meta = np.column_stack((train_pred_logreg, train_pred_rf, train_pred_xgb))

    meta_clf = LogisticRegression(random_state=42)
    meta_clf.fit(X_train_meta, y_train)

    y_val_pred_meta = meta_clf.predict(X_val_meta)
    y_val_prob_meta = meta_clf.predict_proba(X_val_meta)[:,1]
    print("\nEnsemble (Stacking) Validation Metrics:")
    print(classification_report(y_val, y_val_pred_meta))
    print("ROC-AUC:", roc_auc_score(y_val, y_val_prob_meta))

    joblib.dump(meta_clf, "../outputs/stacking_meta_clf.pkl")
    print("Ensemble model saved.")
