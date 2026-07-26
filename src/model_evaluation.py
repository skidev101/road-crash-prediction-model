import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_models(
    models,
    X_test,
    y_test
):

    results = []

    predictions = {}

    for name, model in models.items():

        print(
            f"\nEvaluating {name}..."
        )

        # Predictions

        y_pred = model.predict(
            X_test
        )

        # Probability predictions

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        # Metrics

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        pr_auc = average_precision_score(
            y_test,
            y_probability
        )

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1_Score": f1,

            "ROC_AUC": roc_auc,

            "PR_AUC": pr_auc
        })

        predictions[
            name
        ] = {

            "predictions": y_pred,

            "probabilities": y_probability
        }

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall: {recall:.4f}"
        )

        print(
            f"F1: {f1:.4f}"
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC: {pr_auc:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "outputs/metrics",
        exist_ok=True
    )

    results_df.to_csv(
        "outputs/metrics/model_results.csv",
        index=False
    )

    return (
        results_df,
        predictions
    )


# ============================================================
# PLOT MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    results_df
):

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    results_melted = results_df.melt(

        id_vars="Model",

        value_vars=[
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
            "PR_AUC"
        ],

        var_name="Metric",

        value_name="Score"
    )

    plt.figure(
        figsize=(14, 8)
    )

    for metric in results_melted[
        "Metric"
    ].unique():

        subset = results_melted[
            results_melted["Metric"]
            == metric
        ]

        plt.plot(

            subset["Model"],

            subset["Score"],

            marker="o",

            label=metric
        )

    plt.title(
        "Machine Learning Model Performance Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        rotation=30
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/model_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name
):

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Fatal",
            "Fatal"
        ]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    plt.savefig(
        "outputs/figures/confusion_matrix.png",
        dpi=300
    )

    plt.close()