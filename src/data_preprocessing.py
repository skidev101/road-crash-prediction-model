import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/traffic_accidents.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the original traffic accident dataset.
    """
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully: {df.shape}")

    return df


# ============================================================
# CLEAN AND PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Clean the dataset and create the target variable.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Convert crash_date to datetime
    # --------------------------------------------------------

    df["crash_date"] = pd.to_datetime(
        df["crash_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Extract year from crash date
    # --------------------------------------------------------

    df["crash_year"] = df["crash_date"].dt.year

    # --------------------------------------------------------
    # Create binary target
    #
    # 0 = Non-fatal crash
    # 1 = Fatal crash
    # --------------------------------------------------------

    df["fatal_crash"] = (
        df["injuries_fatal"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"Removed {before - after} duplicate rows."
    )

    # --------------------------------------------------------
    # Display target distribution
    # --------------------------------------------------------

    print("\nTarget distribution:")

    print(
        df["fatal_crash"]
        .value_counts()
    )
    


    print("\nTarget percentage:")

    print(
        df["fatal_crash"]
        .value_counts(normalize=True)
        .mul(100)
        .round(4)
    )

    return df


# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

def get_features_and_target(df):
    """
    Define model features and target variable.

    Injury outcome variables are removed to prevent
    data leakage.
    """

    features = [

        # Environmental conditions
        "traffic_control_device",
        "weather_condition",
        "lighting_condition",

        # Crash and road characteristics
        "first_crash_type",
        "trafficway_type",
        "alignment",
        "roadway_surface_cond",
        "road_defect",

        # Location/context
        "intersection_related_i",

        # Contributing factor
        "prim_contributory_cause",

        # Crash scale
        "num_units",

        # Temporal features
        "crash_hour",
        "crash_day_of_week",
        "crash_month",
        "crash_year"
    ]

    target = "fatal_crash"

    X = df[features].copy()

    y = df[target].copy()

    return X, y, features


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(X, y):
    """
    Split dataset into:

    70% Training
    15% Testing
    15% Validation

    Stratification is used because fatal crashes
    are extremely rare.
    """

    # First split:
    # 70% training
    # 30% temporary

    X_train, X_temp, y_train, y_temp = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=42,

        stratify=y
    )

    # Second split:
    # 15% testing
    # 15% validation

    X_test, X_validation, y_test, y_validation = train_test_split(

        X_temp,
        y_temp,

        test_size=0.50,

        random_state=42,

        stratify=y_temp
    )

    print("\nDataset split:")

    print(
        "Training:",
        X_train.shape
    )

    print(
        "Testing:",
        X_test.shape
    )

    print(
        "Validation:",
        X_validation.shape
    )

    print("\nFatal cases in each split:")

    print(
        "Training:",
        y_train.sum()
    )

    print(
        "Testing:",
        y_test.sum()
    )

    print(
        "Validation:",
        y_validation.sum()
    )

    return (
        X_train,
        X_test,
        X_validation,
        y_train,
        y_test,
        y_validation
    )


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(X):
    """
    Create preprocessing pipeline.

    Categorical features:
    One-hot encoding.

    Numerical features:
    Median imputation.
    """

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print("\nCategorical features:")

    print(categorical_features)

    print("\nNumerical features:")

    print(numerical_features)

    # Categorical pipeline

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Numerical pipeline

    numerical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # Combine pipelines

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),

            (
                "numerical",
                numerical_pipeline,
                numerical_features
            )
        ]
    )

    return preprocessor