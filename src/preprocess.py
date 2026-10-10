from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42

# Text (categorical) columns in the dataset
CAT_COLS = [
    "land_surface_condition", "foundation_type", "roof_type",
    "ground_floor_type", "other_floor_type", "position",
    "plan_configuration", "legal_ownership_status",
]
# Location IDs: numbers that are really categories
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]


def load_data(test_size=0.2):
    """Load the CSVs and return a stratified 80/20 split.

    Returns: X_train, X_val, y_train, y_val
    """
    values = pd.read_csv(DATA_DIR / "train_values.csv", index_col="building_id")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv", index_col="building_id")
    df = values.join(labels)

    X = df.drop(columns="damage_grade")
    y = df["damage_grade"]

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def _numeric_cols(df):
    """Every column that is not categorical and not a geo ID."""
    return [c for c in df.columns if c not in CAT_COLS + GEO_COLS]


def build_preprocessor(kind):
    """Return an unfitted ColumnTransformer.

    kind="linear": for KNN and Neural Network
        - one-hot encode text columns and geo_level_1_id
        - standardize numeric columns
        - drop geo_level_2_id and geo_level_3_id (thousands of values)
    kind="tree": for Random Forest and boosting
        - ordinal-encode text columns
        - keep all other columns as they are (including all geo IDs)
    """
    if kind == "linear":
        return ColumnTransformer(
            [
                ("cat",
                 OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CAT_COLS + ["geo_level_1_id"]),
                ("num", StandardScaler(), _numeric_cols),
            ],
            remainder="drop",
        )
    if kind == "tree":
        return ColumnTransformer(
            [
                ("cat",
                 OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1),
                 CAT_COLS),
            ],
            remainder="passthrough",
        )
    raise ValueError("kind must be 'linear' or 'tree'")


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    print("Split:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    print(y_train.value_counts(normalize=True).round(3))

    for kind in ["linear", "tree"]:
        prep = build_preprocessor(kind)
        Xt = prep.fit_transform(X_train)
        Xv = prep.transform(X_val)
        print(kind, "->", Xt.shape, Xv.shape)