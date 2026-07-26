# Road Crash Fatality Prediction

When a traffic crash occurs, understanding what factors contribute to fatalities can help save lives. This project predicts whether a crash will be fatal using historical accident data, giving safety agencies the insights they need to pinpoint high-risk scenarios and allocate resources more effectively.

A machine learning pipeline built on Chicago traffic accident records. It tackles the extreme class imbalance between fatal and non-fatal crashes, trains multiple models, and picks the best one based on PR-AUC, the most relevant metric for rare-event prediction. The pipeline also visualizes exploratory patterns, ranks risk factors by importance, and saves everything you need to reuse the trained model.

## System Design

```mermaid
flowchart LR
    DataSrc["Traffic Accident Data"]
    Preprocess["Preprocessing Pipeline"]
    SmoteNode["SMOTE Oversampling"]
    Train["Model Training"]
    Eval["Model Evaluation"]
    BestModel["Best Model Selection"]

    DataSrc --> Preprocess
    Preprocess --> SmoteNode
    SmoteNode --> Train
    Train --> Eval
    Eval --> BestModel

    style DataSrc fill:#4c1f24,stroke:#ef4444,stroke-width:2px,color:#fff
    style Preprocess fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style SmoteNode fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#fff
    style Train fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style Eval fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style BestModel fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#fff
```

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:skidev101/road-crash-prediction-model.git
   cd road-crash-prediction-model
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure the data file is placed at `data/traffic_accidents.csv` (the pipeline expects a CSV in that location).

## Usage

Run the entire pipeline with a single command:

```bash
python main.py
```

That's it. The script will:

- Load and clean the dataset
- Create the target variable (`fatal_crash`)
- Split the data into training, testing, and validation sets (70‑15‑15 split, stratified)
- Generate exploratory plots (target distribution, crash types, weather/lighting fatality rates, correlation heatmap)
- Train five models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost
- Evaluate all models and select the best one based on PR‑AUC
- Output model comparison plots, confusion matrix, and feature importance

All results are saved under `outputs/`:

- `outputs/figures/` – all charts (PNG, 300 DPI)
- `outputs/metrics/` – CSV with model performance and feature importance
- `outputs/predictions/` – CSV with test-set predictions and probabilities

The best performing model is serialized to `models/best_fatality_model.pkl` and can be loaded later with:

```python
import joblib
model = joblib.load("models/best_fatality_model.pkl")
```

## Features

### Comprehensive data preprocessing

The pipeline handles missing values, creates a binary target from injury counts, removes duplicates, and separates 10 original categorical features from 5 numerical ones. It then builds a `ColumnTransformer` that one‑hot encodes categories and imputes missing numeric values with the median.

```mermaid
flowchart TD
    Load["Load CSV Dataset"]
    Clean["Clean & Deduplicate"]
    Target["Create Target: fatal_crash"]
    Split["Train/Test/Val Split (Stratified)"]
    Transform["ColumnTransformer: One-Hot + Imputation"]

    Load --> Clean
    Clean --> Target
    Target --> Split
    Split --> Transform

    style Load fill:#4c1f24,stroke:#ef4444,stroke-width:2px,color:#fff
    style Clean fill:#27495c,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Target fill:#27495c,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Split fill:#27495c,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Transform fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#fff
```

### Advanced handling of class imbalance (SMOTE)

Fatal crashes are extremely rare (less than 1% of records). To give the models a fighting chance, the pipeline applies SMOTE (Synthetic Minority Over‑sampling Technique) after preprocessing. This generates synthetic examples of the minority class, balancing the training set without simply duplicating original data.

```mermaid
sequenceDiagram
    actor Pipeline
    participant Preprocessor
    participant SMOTE
    participant Trainer

    Pipeline->>Preprocessor: transform training features
    Preprocessor-->>Pipeline: one-hot encoded matrix
    Pipeline->>SMOTE: fit_resample(X, y)
    SMOTE-->>Pipeline: resampled balanced dataset
    Pipeline->>Trainer: fit on resampled data
```

### Multi-model training and PR‑AUC optimization

Five different classifiers are trained, including both linear and tree‑based models. Because accuracy is misleading with such a severe class imbalance, the pipeline selects the best model using Precision‑Recall AUC, a metric that focuses on the minority class and rewards models that can find true fatalities while limiting false alarms.

```mermaid
flowchart LR
    Models["Train 5 Models"]
    Evaluate["Evaluate on Test Set"]
    Compare["Compare PR‑AUC Scores"]
    Select["Select Best Model"]
    Save["Save to model.pkl"]

    Models --> Evaluate
    Evaluate --> Compare
    Compare --> Select
    Select --> Save

    style Models fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style Evaluate fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#fff
    style Compare fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#fff
    style Select fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Save fill:#4c1f24,stroke:#ef4444,stroke-width:2px,color:#fff
```

### Explainability: feature importance ranking

After the best model is chosen (typically the tree‑based ensemble that performed best on PR‑AUC), the pipeline extracts and plots the top risk factors. This tells you exactly which features drive the prediction, making the results actionable for traffic safety analysis.

## Technologies Used

| Technology           | Purpose                                      |
| -------------------- | -------------------------------------------- |
| Python               | Core programming language                    |
| NumPy                | Numerical computations                       |
| pandas               | Data manipulation and analysis               |
| scikit-learn         | Preprocessing, train/test split, base models |
| XGBoost              | Gradient boosting classifier                 |
| imbalanced‑learn     | SMOTE for class imbalance handling           |
| matplotlib / seaborn | Visualizations and exploratory analysis      |
| joblib               | Model serialization                          |

## Author

- GitHub: [https://github.com/skidev101](https://github.com/skidev101)

---

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-549F5A?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://imbalanced-learn.org/)
