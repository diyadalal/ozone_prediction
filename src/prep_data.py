"""
Data loading, cleaning and splitting for the ozone (8hr) data.
Use for each model created to prep data.
"""

import pandas as pd

ID_COLS = ["Dataset", "Date"]
TARGET = "Class"
DEFAULT_PATH = "../data/ozone_eighthr.csv"


def load_data(path=DEFAULT_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def clean(df, max_missing=0.10, verbose=True):
    """ Handles missing data and removes features with too many missing values.
    1. Drop feature columns with more than 10% missing.
    2. Drop any remaining rows that still contain a null.
    """
    features = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    miss = df[features].isna().mean()
    drop_cols = miss[miss > max_missing].index.tolist()
    features = [c for c in features if c not in drop_cols]

    out = df.drop(columns=drop_cols)
    n_before = len(out)
    out = out.dropna(subset=features).reset_index(drop=True)

    if verbose:
        print(f"Dropped {len(drop_cols)} columns: {drop_cols}")
        print(f"Dropped {n_before - len(out)} rows ({(n_before - len(out)) / n_before:.1%}); "
              f"{len(out)} rows and {len(features)} features remain")
    return out, features


def time_split(df, train_frac=0.70, val_frac=0.15):
    """ Chronological train/validation/test split .
    Oldest data for train, newest for test.
    """
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def get_splits(path=DEFAULT_PATH, max_missing=0.10, train_frac=0.70, val_frac=0.15, verbose=True):
    """ Call this method to load data, clean it, and split it when making models.
    Returns a dict with X/y for each split plus the feature list.
    keys: X_train, y_train, X_val, y_val, X_test, y_test, features, clean
    """
    df = load_data(path)
    clean_df, features = clean(df, max_missing, verbose)
    train, val, test = time_split(clean_df, train_frac, val_frac)

    if verbose:
        for name, d in [("train", train), ("val", val), ("test", test)]:
            print(f"{name:5s} {len(d):5d} rows  {d.Date.min().date()} -> {d.Date.max().date()}  "
                  f"ozone-day rate: {d[TARGET].mean():.3f}")

    return {
        "X_train": train[features], "y_train": train[TARGET],
        "X_val": val[features], "y_val": val[TARGET],
        "X_test": test[features], "y_test": test[TARGET],
        "features": features, "clean": clean_df,
    }