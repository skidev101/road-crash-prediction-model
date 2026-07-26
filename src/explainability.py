import os

import pandas as pd

import matplotlib.pyplot as plt


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    model_name
):

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    # Get preprocessor

    preprocessor = model[
        "preprocessor"
    ]

    # Get trained model

    trained_model = model[
        "model"
    ]

    # Get feature names

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Check whether model supports feature importance

    if not hasattr(
        trained_model,
        "feature_importances_"
    ):

        print(
            f"{model_name} does not provide "
            "feature_importances_."
        )

        return None

    importance = (
        trained_model
        .feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    # Save results

    importance_df.to_csv(

        "outputs/metrics/"
        "feature_importance.csv",

        index=False
    )

    # Plot top 20

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(

        top_features["Feature"],

        top_features["Importance"]
    )

    plt.title(
        f"Top Risk Factors - {model_name}"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(

        "outputs/figures/"
        "feature_importance.png",

        dpi=300
    )

    plt.close()

    return importance_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    X_test,
    y_test,
    predictions,
    model_name
):

    os.makedirs(
        "outputs/predictions",
        exist_ok=True
    )

    prediction_data = X_test.copy()

    prediction_data[
        "actual_fatal_crash"
    ] = y_test.values

    prediction_data[
        "predicted_fatal_crash"
    ] = predictions[
        "predictions"
    ]

    prediction_data[
        "fatality_probability"
    ] = predictions[
        "probabilities"
    ]

    prediction_data.to_csv(

        "outputs/predictions/"
        "fatality_predictions.csv",

        index=False
    )