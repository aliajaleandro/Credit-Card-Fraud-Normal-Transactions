import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "creditcard.csv"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def generate_eda_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Class", hue="Class", dodge=False, legend=False)
    ax.set_title("Shperndarja e klasave")
    ax.set_xlabel("Klasa (0 = normale, 1 = mashtrim)")
    ax.set_ylabel("Numri i transaksioneve")
    save_figure(IMAGES_DIR / "class_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Amount"], bins=60, kde=True, color="#1f77b4")
    plt.title("Shperndarja e vleres se transaksioneve")
    plt.xlabel("Amount")
    plt.ylabel("Frekuenca")
    save_figure(IMAGES_DIR / "amount_distribution.png")

    sampled = df.sample(n=5000, random_state=42)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sampled,
        x="V14",
        y="V17",
        hue="Class",
        alpha=0.7,
        palette={0: "#4c78a8", 1: "#e45756"},
    )
    plt.title("Scatter plot i V14 dhe V17")
    plt.xlabel("V14")
    plt.ylabel("V17")
    save_figure(IMAGES_DIR / "scatter_v14_v17.png")

    corr = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False)
    top_features = corr.head(10).index.tolist() + ["Class"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix per atributet me te lidhura me klasen")
    save_figure(IMAGES_DIR / "correlation_matrix_top_features.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df.sample(n=10000, random_state=42), x="Class", y="Amount", hue="Class", dodge=False, legend=False)
    plt.title("Boxplot i Amount sipas klases")
    plt.xlabel("Klasa")
    plt.ylabel("Amount")
    save_figure(IMAGES_DIR / "amount_boxplot_by_class.png")


def build_modeling_sample(df: pd.DataFrame) -> pd.DataFrame:
    fraud_df = df[df["Class"] == 1]
    non_fraud_df = df[df["Class"] == 0].sample(n=40000, random_state=42)
    modeled_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True).sample(frac=1, random_state=42)
    return modeled_df


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_models(scale_pos_weight: float):
    numeric_features = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

    linear_preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_features)],
        remainder="drop",
    )

    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=40,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
            random_state=42,
        ),
    }

    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    metrics_rows = []
    reports = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        if name == "Linear Regression":
            y_score = np.clip(model.predict(X_test), 0, 1)
            y_pred = (y_score >= 0.5).astype(int)
        else:
            y_score = model.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_score),
            "PR-AUC": average_precision_score(y_test, y_score),
        }
        metrics_rows.append(metrics)
        reports[name] = classification_report(y_test, y_pred, digits=4)
        roc_data[name] = roc_curve(y_test, y_score)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix - {name}")
        save_figure(IMAGES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="F1-Score", ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    with open(RESULTS_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        for model_name, report in reports.items():
            f.write(f"{model_name}\n")
            f.write(report)
            f.write("\n" + "=" * 80 + "\n\n")

    with open(RESULTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve per krahasimin e modeleve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_figure(IMAGES_DIR / "roc_curves.png")

    chart_df = metrics_df.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-Score", "ROC-AUC"])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=chart_df, x="Model", y="value", hue="variable")
    plt.title("Krahasimi i metrikave kryesore")
    plt.xlabel("Modeli")
    plt.ylabel("Vlera")
    plt.ylim(0, 1.05)
    save_figure(IMAGES_DIR / "model_metric_comparison.png")

    return metrics_df


def save_dataset_summary(df: pd.DataFrame, modeled_df: pd.DataFrame) -> None:
    summary = {
        "full_dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "fraud_cases": int(df["Class"].sum()),
            "non_fraud_cases": int((df["Class"] == 0).sum()),
            "fraud_ratio_percent": round(float(df["Class"].mean() * 100), 4),
            "amount_mean": round(float(df["Amount"].mean()), 4),
            "amount_median": round(float(df["Amount"].median()), 4),
        },
        "modeling_sample": {
            "rows": int(modeled_df.shape[0]),
            "fraud_cases": int(modeled_df["Class"].sum()),
            "non_fraud_cases": int((modeled_df["Class"] == 0).sum()),
            "fraud_ratio_percent": round(float(modeled_df["Class"].mean() * 100), 4),
        },
    }

    with open(RESULTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    df = load_data()
    generate_eda_plots(df)

    modeled_df = build_modeling_sample(df)
    save_dataset_summary(df, modeled_df)

    X_train, X_test, y_train, y_test = prepare_data(modeled_df)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = build_models(scale_pos_weight=scale_pos_weight)
    metrics_df = evaluate_models(models, X_train, X_test, y_train, y_test)

    print("Analiza perfundoi me sukses.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
