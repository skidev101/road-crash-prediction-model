# ============================================================
# MAIN MACHINE LEARNING PIPELINE
# ============================================================

from src.data_preprocessing import (
    load_data,
    prepare_data,
    get_features_and_target,
    split_data,
    create_preprocessor
)

from src.exploratory_analysis import (
    plot_target_distribution,
    plot_crash_types,
    plot_weather_fatality,
    plot_lighting_fatality,
    plot_correlation_heatmap
)

from src.model_training import (
    train_models,
    save_best_model
)

from src.model_evaluation import (
    evaluate_models,
    plot_model_comparison,
    plot_confusion_matrix
)

from src.explainability import (
    get_feature_importance,
    save_predictions
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# 2. PREPARE DATA
# ============================================================

df = prepare_data(
    df
)


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print(
    "\nGenerating exploratory analysis..."
)

plot_target_distribution(
    df
)

plot_crash_types(
    df
)

plot_weather_fatality(
    df
)

plot_lighting_fatality(
    df
)

plot_correlation_heatmap(
    df
)


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

X, y, features = (
    get_features_and_target(
        df
    )
)


print(
    "\nNumber of features:",
    len(features)
)

print(
    "Feature columns:"
)

for feature in features:

    print("-", feature)


# ============================================================
# 5. SPLIT DATA
# ============================================================

(
    X_train,
    X_test,
    X_validation,
    y_train,
    y_test,
    y_validation
) = split_data(
    X,
    y
)


# ============================================================
# 6. CREATE PREPROCESSOR
# ============================================================

preprocessor = create_preprocessor(
    X_train
)


# ============================================================
# 7. TRAIN MODELS
# ============================================================

print(
    "\nStarting model training..."
)

models = train_models(

    preprocessor,

    X_train,

    y_train
)


# ============================================================
# 8. EVALUATE MODELS ON TEST SET
# ============================================================

print(
    "\nEvaluating models..."
)

(
    results_df,
    predictions
) = evaluate_models(

    models,

    X_test,

    y_test
)


# ============================================================
# 9. DISPLAY MODEL COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "MODEL PERFORMANCE COMPARISON"
)

print(
    "=" * 60
)

print(
    results_df
    .sort_values(
        "PR_AUC",
        ascending=False
    )
)


# ============================================================
# 10. PLOT MODEL COMPARISON
# ============================================================

plot_model_comparison(
    results_df
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

# For highly imbalanced data,
# PR-AUC is our primary selection metric.

best_model_name = (
    results_df
    .sort_values(
        "PR_AUC",
        ascending=False
    )
    .iloc[0]["Model"]
)

best_model = models[
    best_model_name
]


print(
    f"\nBest model based on PR-AUC: "
    f"{best_model_name}"
)

# Save the best model
save_best_model(
    best_model,
    best_model_name
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

plot_confusion_matrix(

    best_model,

    X_test,

    y_test,

    best_model_name
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance_df = (
    get_feature_importance(

        best_model,

        best_model_name
    )
)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

save_predictions(

    X_test,

    y_test,

    predictions[
        best_model_name
    ],

    best_model_name
)


# ============================================================
# 15. FINAL VALIDATION
# ============================================================

print(
    "\nEvaluating final model "
    "on validation dataset..."
)

validation_pred = (
    best_model
    .predict(
        X_validation
    )
)

validation_probability = (
    best_model
    .predict_proba(
        X_validation
    )[:, 1]
)


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


print(
    "\nFinal Validation Performance:"
)

print(
    "Precision:",
    precision_score(
        y_validation,
        validation_pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_validation,
        validation_pred,
        zero_division=0
    )
)

print(
    "F1 Score:",
    f1_score(
        y_validation,
        validation_pred,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_validation,
        validation_probability
    )
)

print(
    "PR-AUC:",
    average_precision_score(
        y_validation,
        validation_probability
    )
)


print(
    "\n"
    + "=" * 60
)

print(
    "PROJECT PIPELINE COMPLETED SUCCESSFULLY!"
)

print(
    "=" * 60
)