import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    # Load preprocessed data
    X_train = pd.read_pickle("../data/processed/X_train.pkl")
    y_train = np.load("../data/processed/y_train.npy")
    X_val = pd.read_pickle("../data/processed/X_val.pkl")
    y_val = np.load("../data/processed/y_val.npy")
    X_test = pd.read_pickle("../data/processed/X_test.pkl")
    y_test = np.load("../data/processed/y_test.npy")

    # Apply SMOTE only on training data
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    # Save the resampled training set
    X_train_resampled.to_pickle("../data/processed/X_train_resampled.pkl")
    np.save("../data/processed/y_train_resampled.npy", y_train_resampled)

    print("Class imbalance handled with SMOTE. Resampled data saved.")
