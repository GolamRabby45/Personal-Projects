import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

def train_xgboost(X_train, y_train, X_val, y_val, max_rounds=100, early_stopping_rounds=10):
    """
    Train an XGBoost model with manual early stopping.
    """
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    best_model = None
    best_score = float('inf')
    no_improve_rounds = 0

    for round_num in range(1, max_rounds + 1):
        xgb.n_estimators = round_num
        xgb.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = xgb.predict_proba(X_val)[:, 1]
        logloss = -np.mean(y_val * np.log(y_val_pred) + (1 - y_val) * np.log(1 - y_val_pred))

        # Check for improvement
        if logloss < best_score:
            best_score = logloss
            best_model = joblib.dump(xgb, "../outputs/best_xgb_model.pkl")  # Save the best model
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        # Early stopping check
        if no_improve_rounds >= early_stopping_rounds:
            print(f"Early stopping at round {round_num}. Best logloss: {best_score:.4f}")
            break

    # Load the best model back
    xgb = joblib.load("../outputs/best_xgb_model.pkl")

    # Evaluate the final model
    y_val_pred = xgb.predict(X_val)
    y_val_prob = xgb.predict_proba(X_val)[:, 1]
    print("XGBoost Validation Metrics:")
    print(classification_report(y_val, y_val_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_val_prob))

    # Save the final model
    joblib.dump(xgb, "../outputs/xgb_model_final.pkl")

def train_autoencoder(X_train, X_val, y_val, encoding_dim=14, epochs=10, batch_size=256):
    """
    Train an autoencoder for unsupervised anomaly detection.
    """
    X_train_norm = X_train[y_train == 0]
    
    input_dim = X_train_norm.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu', 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(X_train_norm, X_train_norm,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_val, X_val))

    X_val_pred = autoencoder.predict(X_val)
    reconstruction_error = np.mean(np.power(X_val - X_val_pred, 2), axis=1)

    threshold = np.percentile(reconstruction_error, 99)
    y_val_pred_ae = (reconstruction_error > threshold).astype(int)

    print("Autoencoder (Unsupervised) Validation Metrics:")
    print(classification_report(y_val, y_val_pred_ae))
    print("ROC-AUC:", roc_auc_score(y_val, reconstruction_error))

    autoencoder.save("../outputs/autoencoder_model.h5")
    np.save("../outputs/autoencoder_threshold.npy", np.array([threshold]))

if __name__ == "__main__":
    # Load processed and resampled data for supervised methods
    X_train = pd.read_pickle("../data/processed/X_train_resampled.pkl")
    y_train = np.load("../data/processed/y_train_resampled.npy")
    X_val = pd.read_pickle("../data/processed/X_val.pkl")
    y_val = np.load("../data/processed/y_val.npy")
    X_test = pd.read_pickle("../data/processed/X_test.pkl")
    y_test = np.load("../data/processed/y_test.npy")

    # Train XGBoost
    train_xgboost(X_train, y_train, X_val, y_val)

    # Train Autoencoder for unsupervised anomaly detection
    train_autoencoder(X_train, X_val, y_val, encoding_dim=14, epochs=5, batch_size=256)
