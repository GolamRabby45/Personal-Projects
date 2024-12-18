import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(df: pd.DataFrame, target: str, logger, model_type="classification", test_size=0.2, random_state=42):
    if target not in df.columns:
        logger.error(f"Target column {target} not found in dataset.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    # Handle categorical variables if needed (for demonstration, assume all numeric or already encoded)
    # If categorical encoding is needed, do it here.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Split data into train/test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    if model_type == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Classification accuracy: {acc:.4f}")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
    else:
        # For regression, you could use LinearRegression and MSE/R2
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Regression MSE: {mse:.4f}, R2: {r2:.4f}")

    return model
