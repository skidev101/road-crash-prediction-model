import os

import joblib

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    preprocessor,
    X_train,
    y_train
):

    models = {}

    # --------------------------------------------------------
    # 1. Logistic Regression
    # --------------------------------------------------------

    models[
        "Logistic Regression"
    ] = Pipeline(

        steps=[
            ("preprocessor", preprocessor),

            ("smote", SMOTE(random_state=42)),

            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]
        
    )

    # --------------------------------------------------------
    # 2. Decision Tree
    # --------------------------------------------------------

    models[
        "Decision Tree"
    ] = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            ("smote", SMOTE(random_state=42)),

            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=10,
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 3. Random Forest
    # --------------------------------------------------------

    models[
        "Random Forest"
    ] = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            ("smote", SMOTE(random_state=42)),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 4. Gradient Boosting
    # --------------------------------------------------------

    models[
        "Gradient Boosting"
    ] = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            ("smote", SMOTE(random_state=42)),

            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 5. XGBoost
    # --------------------------------------------------------

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    models[
        "XGBoost"
    ] = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            ("smote", SMOTE(random_state=42)),

            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    # scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    trained_models = {}

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        preprocessor = model.named_steps["preprocessor"]
        smote = model.named_steps["smote"]

        X_processed = preprocessor.fit_transform(X_train)

        X_resampled, y_resampled = smote.fit_resample(
            X_processed,
            y_train
        )

        print("\nBefore SMOTE:")
        print(y_train.value_counts())

        print("\nAfter SMOTE:")
        print(y_resampled.value_counts())

        print(y_train.value_counts())
        print(y_resampled.value_counts())

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        print(
            f"{name} training completed."
        )

    return trained_models

# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(model, model_name):

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = "models/best_fatality_model.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nBest model saved successfully!"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Location: {model_path}"
    )