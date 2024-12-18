import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(data_path):
    return pd.read_csv(data_path)

def feature_engineering(df):
    # The 'Time' feature is the seconds elapsed between this transaction and the first transaction in the dataset.
    # We can extract an hour-of-day feature as a simple transformation:
    # Since we don't have actual day info, we assume a 24-hour cycle for demonstration.
    df['Hour'] = (df['Time'] / 3600) % 24
    return df

def scale_features(train, val, test, features_to_scale):
    scaler = StandardScaler()
    # Fit on train only
    scaler.fit(train[features_to_scale])
    # Transform on train, val, test
    train[features_to_scale] = scaler.transform(train[features_to_scale])
    val[features_to_scale] = scaler.transform(val[features_to_scale])
    test[features_to_scale] = scaler.transform(test[features_to_scale])
    return train, val, test

if __name__ == "__main__":
    data_path = "../data/creditcard.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file not found. Please ensure creditcard.csv is in the data directory.")

    df = load_data(data_path)
    
    # Feature Engineering
    df = feature_engineering(df)

    # Split data: We'll first separate features and labels
    X = df.drop('Class', axis=1)
    y = df['Class'].values

    # Train/Val/Test split: 
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                                test_size=0.2, 
                                                                random_state=42, 
                                                                stratify=y)
    # Then split train_val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.25, 
                                                      random_state=42, 
                                                      stratify=y_train_val)
    # Now we have 60% train, 20% val, 20% test

    # Scale selected features
    # 'Amount' and 'Hour' might need scaling. The PCA components and 'Time' are less interpretable, but we could scale them too.
    # We'll scale 'Amount' and 'Hour'.
    features_to_scale = ['Amount', 'Hour']
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test, features_to_scale)

    # Save processed sets
    os.makedirs("../data/processed", exist_ok=True)
    X_train.to_pickle("../data/processed/X_train.pkl")
    X_val.to_pickle("../data/processed/X_val.pkl")
    X_test.to_pickle("../data/processed/X_test.pkl")
    np.save("../data/processed/y_train.npy", y_train)
    np.save("../data/processed/y_val.npy", y_val)
    np.save("../data/processed/y_test.npy", y_test)

    print("Preprocessing complete. Train/Val/Test sets saved in ../data/processed/")
