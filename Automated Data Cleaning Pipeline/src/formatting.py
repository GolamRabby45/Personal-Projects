import pandas as pd

def format_dates(df: pd.DataFrame, date_cols: list, logger):
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted {col} to datetime.")
            except Exception as e:
                logger.error(f"Failed to convert {col} to datetime: {e}")
        else:
            logger.warning(f"{col} not found for date formatting.")
    return df

def clean_categorical(df: pd.DataFrame, cat_cfg: dict, logger):
    cols = cat_cfg.get("columns", [])
    lowercase = cat_cfg.get("lowercase", False)
    strip = cat_cfg.get("strip", False)

    for col in cols:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            original_unique = df[col].unique()
            series = df[col]
            if lowercase:
                series = series.str.lower()
            if strip:
                series = series.str.strip()
            df[col] = series
            logger.info(f"Cleaned categorical column {col}. Original unique values: {original_unique}, New unique values: {df[col].unique()}")
        else:
            logger.warning(f"{col} not found or not string type for categorical cleaning.")
    return df

def round_numeric(df: pd.DataFrame, round_cfg: dict, logger):
    cols = round_cfg.get("columns", [])
    decimals = round_cfg.get("decimals", 2)

    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(decimals)
            logger.info(f"Rounded numeric column {col} to {decimals} decimals.")
        else:
            logger.warning(f"{col} not found or not numeric for rounding.")
    return df

def apply_formatting(df: pd.DataFrame, config: dict, logger):
    fmt_cfg = config.get("formatting", {})

    # Date formatting
    date_cols = fmt_cfg.get("date_columns", [])
    df = format_dates(df, date_cols, logger)

    # Categorical cleaning
    cat_cfg = fmt_cfg.get("categorical_cleaning", {})
    df = clean_categorical(df, cat_cfg, logger)

    # Numeric rounding
    round_cfg = fmt_cfg.get("numeric_rounding", {})
    df = round_numeric(df, round_cfg, logger)

    return df
