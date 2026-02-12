# src/recommender/pipeline.py
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


CAT_COLS = ["ClientCountry", "ClientOptINEmail", "ClientOptINPhone", "ClientGender"]
NUM_COLS = ["Age"]


def make_preprocess_pipeline() -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
