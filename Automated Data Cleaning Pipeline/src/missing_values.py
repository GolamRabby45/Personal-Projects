import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def handle_missing_values(df: pd.DataFrame, strategy_map: dict, knn_settings: dict, logger):
    """
    Handles missing values according to the strategy_map.
    strategy_map: dict of {column: strategy}, where strategy in ['drop', 'mean', 'median', 'mode', 'knn']
    knn_settings: dict with parameters for KNNImputer if 'knn' strategy is used.
    """

    # Separate columns by strategy
    drop_cols = [col for col, strat in strategy_map.items() if strat == 'drop']
    mean_cols = [col for col, strat in strategy_map.items() if strat == 'mean']
    median_cols = [col for col, strat in strategy_map.items() if strat == 'median']
    mode_cols = [col for col, strat in strategy_map.items() if strat == 'mode']
    knn_cols = [col for col, strat in strategy_map.items() if strat == 'knn']

    # Handle drop: If configured to drop rows where a certain col is missing
    for col in drop_cols:
        before_count = len(df)
        df = df.dropna(subset=[col])
        after_count = len(df)
        logger.info(f"Dropped rows with missing {col}. Rows before: {before_count}, after: {after_count}")

    # For mean, median, mode, we can do simple imputations
    for col in mean_cols:
        if df[col].isna().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            logger.info(f"Imputed missing {col} with mean: {mean_val}")

    for col in median_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Imputed missing {col} with median: {median_val}")

    for col in mode_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode().iloc[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"Imputed missing {col} with mode: {mode_val}")

    # If KNN strategy is chosen for some columns, we apply KNNImputer.
    # KNNImputer works on numeric data. If categorical columns are KNN-imputed,
    # you should encode them or skip KNN for them.
    if knn_cols:
        # We will apply KNNImputer to the subset of columns that need KNN plus any needed numeric cols.
        # For simplicity, let's just apply it to knn_cols. They must be numeric.
        knn_data = df[knn_cols]
        # Ensure numeric
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in knn_cols):
            raise ValueError("KNN imputation requires all columns to be numeric.")
        
        imputer = KNNImputer(n_neighbors=knn_settings.get("n_neighbors", 5))
        imputed_arr = imputer.fit_transform(knn_data)
        df[knn_cols] = imputed_arr
        logger.info(f"KNN imputation applied to columns: {knn_cols}")

    return df
