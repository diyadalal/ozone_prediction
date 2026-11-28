"""
Data loading, cleaning and splitting for the ozone (8hr) data.
Use for each model created to prep data.

Task: NEXT-DAY prediction. Each row holds one day's weather and the target is
whether the FOLLOWING calendar day is an ozone day. (horizon="same_day" keeps the
old same-day setup for comparison.)

Missing values are NOT filled in here. Use make_model_pipeline() so the imputer is
fit on training data only (no leakage from validation/test).
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLS = ["Dataset", "Date"]
CLASS = "Class"            # ozone day label for the row's own date
TARGET = "target"          # label the models predict (next day by default)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "ozone_eighthr.csv"


def load_data(path=DEFAULT_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS + [CLASS, TARGET]]


def add_target(df, horizon="next_day", include_today_class=False, verbose=True):
    """ Builds the TARGET column.
    next_day: target = Class of the next calendar day. Rows whose next calendar day
              is missing from the data are dropped (can't label them).
    same_day: target = Class of the same day.
    include_today_class: add today's ozone label as a feature ("Class_today").
              Today's peak is known by evening, so this is legal for next-day only.
    """
    out = df.copy()
    if horizon == "same_day":
        out[TARGET] = out[CLASS]
        return out
    if horizon != "next_day":
        raise ValueError("horizon must be 'next_day' or 'same_day'")

    next_date = out["Date"].shift(-1)
    has_next = (next_date - out["Date"]).dt.days == 1
    out[TARGET] = out[CLASS].shift(-1)
    if include_today_class:
        out["Class_today"] = out[CLASS]
    n_before = len(out)
    out = out[has_next].reset_index(drop=True)
    out[TARGET] = out[TARGET].astype(int)
    if verbose:
        print(f"Next-day target: dropped {n_before - len(out)} rows with no following calendar day")
    return out


def clean(df, max_row_missing=0.50, verbose=True):
    """ Drops days that are mostly empty (sensor outages).
    Rows missing more than max_row_missing of their features are removed; any remaining
    gaps are filled in later by the model pipeline's imputer.
    """
    features = feature_columns(df)
    row_missing = df[features].isna().mean(axis=1)
    keep = row_missing <= max_row_missing
    out = df[keep].reset_index(drop=True)

    if verbose:
        n_drop = (~keep).sum()
        print(f"Dropped {n_drop} rows ({n_drop / len(df):.1%}) with >{max_row_missing:.0%} features missing "
              f"({int(df.loc[~keep, TARGET].sum())} were positive); "
              f"{len(out)} rows and {len(features)} features remain")
        print(f"Remaining cells missing (to be imputed): {out[features].isna().mean().mean():.1%}")
    return out, features


def time_split(df, train_frac=0.70, val_frac=0.15):
    """ Chronological train/validation/test split.
    Oldest data for train, newest for test.
    """
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def get_splits(path=DEFAULT_PATH, horizon="next_day", include_today_class=False,
               max_row_missing=0.50, train_frac=0.70, val_frac=0.15, verbose=True):
    """ Call this method to load data, build the target, clean it, and split it.
    Returns a dict with X/y for each split plus the feature list.
    keys: X_train, y_train, X_val, y_val, X_test, y_test, features, clean
    X may contain NaNs -> always fit models with make_model_pipeline().
    """
    df = add_target(load_data(path), horizon, include_today_class, verbose)
    clean_df, features = clean(df, max_row_missing, verbose)
    train, val, test = time_split(clean_df, train_frac, val_frac)

    if verbose:
        for name, d in [("train", train), ("val", val), ("test", test)]:
            print(f"{name:5s} {len(d):5d} rows  {d.Date.min().date()} -> {d.Date.max().date()}  "
                  f"positives: {int(d[TARGET].sum()):3d}  rate: {d[TARGET].mean():.3f}")

    return {
        "X_train": train[features], "y_train": train[TARGET],
        "X_val": val[features], "y_val": val[TARGET],
        "X_test": test[features], "y_test": test[TARGET],
        "features": features, "clean": clean_df,
    }


def make_model_pipeline(clf, scale=True, impute_strategy="median"):
    """ Imputer (+ optional scaler) + classifier. Use for every model.
    scale=True for linear models; tree models (RF, GB) don't need it.
    """
    steps = [("impute", SimpleImputer(strategy=impute_strategy))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)
