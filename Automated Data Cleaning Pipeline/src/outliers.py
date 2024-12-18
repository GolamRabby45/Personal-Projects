import pandas as pd
from sklearn.ensemble import IsolationForest

def iqr_outliers(df: pd.DataFrame, columns: list, multiplier: float, action: str, logger):
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for outlier detection.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric; skipping outlier detection.")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()

        logger.info(f"Detected {outlier_count} outliers in {col} using IQR method.")

        if action == "remove":
            before = len(df)
            df = df[~outliers_mask]
            after = len(df)
            logger.info(f"Removed {before - after} outliers from {col}.")
        elif action == "cap":
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            logger.info(f"Capped outliers in {col} at [{lower_bound}, {upper_bound}].")
        elif action == "flag":
            # Create a flag column indicating outliers
            flag_col = col + "_outlier_flag"
            df[flag_col] = outliers_mask.astype(int)
            logger.info(f"Flagged outliers in {col} with column {flag_col}.")

    return df

def isolation_forest_outliers(df: pd.DataFrame, columns: list, contamination: float, action: str, logger):
    # Isolation Forest requires numeric data.
    # Apply only to numeric columns specified.
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        logger.warning("No numeric columns for Isolation Forest outlier detection.")
        return df

    iforest = IsolationForest(contamination=contamination, random_state=42)
    subset = df[numeric_cols].values
    preds = iforest.fit_predict(subset)  # -1 for outliers, 1 for inliers

    outliers_mask = (preds == -1)
    outlier_count = outliers_mask.sum()
    logger.info(f"Isolation Forest detected {outlier_count} outliers in columns: {numeric_cols}.")

    if action == "remove":
        before = len(df)
        df = df[~outliers_mask]
        after = len(df)
        logger.info(f"Removed {before - after} outliers based on Isolation Forest.")
    elif action == "flag":
        df["isolation_forest_outlier_flag"] = outliers_mask.astype(int)
        logger.info("Flagged isolation forest outliers in 'isolation_forest_outlier_flag' column.")
    else:
        logger.warning("Only 'remove' or 'flag' actions supported for isolation forest in this demo.")

    return df

def handle_outliers(df: pd.DataFrame, config: dict, logger):
    outlier_cfg = config.get("outlier_handling", {})
    method = outlier_cfg.get("method", "iqr")
    columns = outlier_cfg.get("columns", [])
    action = outlier_cfg.get("action", "remove")

    if method == "iqr":
        multiplier = outlier_cfg.get("iqr_multiplier", 1.5)
        df = iqr_outliers(df, columns, multiplier, action, logger)
    elif method == "isolation_forest":
        iso_cfg = outlier_cfg.get("isolation_forest", {})
        contamination = iso_cfg.get("contamination", 0.01)
        df = isolation_forest_outliers(df, columns, contamination, action, logger)
    else:
        logger.info("No outlier handling method specified or recognized.")
    return df
