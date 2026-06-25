from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

BASE = Path("poi-date-7.29(1)") / "poi-date-7.29"
OUT = Path("outputs") / "random_rf_figures"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORY_FILES = {
    "CAT": BASE / "CAT_random.csv",
    "ISS": BASE / "ISS_random.csv",
    "LEX": BASE / "LEX_random.csv",
    "RTL": BASE / "RTL_random.csv",
}

FEATURES = ["Bea", "Saf", "Act", "Wth", "Dpr", "Bro"]
FEATURE_LABELS = {
    "Bea": "Aesthetics",
    "Saf": "Safety",
    "Act": "Vibrancy",
    "Wth": "Wealth",
    "Dpr": "Depression",
    "Bro": "Boring",
}

RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": None,
    "max_features": "log2",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

PANEL_TITLES = {
    "CAT": "(a) CAT",
    "ISS": "(b) ISS",
    "LEX": "(c) LEX",
    "RTL": "(d) RTL",
}


def read_model_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gb18030", low_memory=False)
    model_df = df[["star"] + FEATURES].copy()
    for col in ["star"] + FEATURES:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna(subset=["star"] + FEATURES)


def fit_models():
    models = {}
    test_data = {}
    shap_data = {}
    metric_rows = []
    importance_rows = []

    for code, path in CATEGORY_FILES.items():
        data = read_model_data(path)
        x = data[FEATURES]
        y = data["star"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42
        )

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        models[code] = model
        test_data[code] = (x_test.reset_index(drop=True), y_test.reset_index(drop=True))

        metric_rows.append(
            {
                "category": code,
                "rows_used": len(data),
                "train_n": len(x_train),
                "test_n": len(x_test),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
            }
        )

        for feature, value in zip(FEATURES, model.feature_importances_):
            importance_rows.append(
                {"category": code, "feature": feature, "rf_importance": value}
            )

        sample_n = min(500, len(x_test))
        sample_x = x_test.sample(sample_n, random_state=42).reset_index(drop=True)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_x, check_additivity=False)
        shap_data[code] = {
            "x": sample_x,
            "values": np.asarray(shap_values),
            "explainer": explainer,
        }

    pd.DataFrame(metric_rows).to_csv(OUT / "random_rf_metrics.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        OUT / "random_rf_feature_importance.csv", index=False
    )
    with open(OUT / "random_rf_params.json", "w", encoding="utf-8") as f:
        json.dump(RF_PARAMS, f, ensure_ascii=False, indent=2, default=str)
    return models, test_data, shap_data, pd.DataFrame(metric_rows)


def plot_shap_summary(shap_data):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=False)
    axes = axes.ravel()
    rng = np.random.default_rng(42)

    for ax, code in zip(axes, CATEGORY_FILES):
        x = shap_data[code]["x"]
        values = shap_data[code]["values"]
        order = np.argsort(np.abs(values).mean(axis=0))[::-1]

        for y_pos, feature_idx in enumerate(order):
            feature = FEATURES[feature_idx]
            shap_v = values[:, feature_idx]
            feat_v = x.iloc[:, feature_idx].to_numpy()
            denom = np.nanmax(feat_v) - np.nanmin(feat_v)
            norm_v = (feat_v - np.nanmin(feat_v)) / denom if denom else np.zeros_like(feat_v)
            jitter = rng.normal(0, 0.075, size=len(shap_v))
            ax.scatter(
                shap_v,
                np.full(len(shap_v), y_pos) + jitter,
                c=norm_v,
                cmap="coolwarm",
                s=8,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )

        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FEATURE_LABELS[FEATURES[i]] for i in order])
        ax.invert_yaxis()
        ax.set_title(PANEL_TITLES[code], loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("SHAP value")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.72, pad=0.02)
    cbar.set_label("Feature value (low to high)")
    fig.suptitle("Fig. 5 Summary chart of feature importance SHAP values", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.96, 0.97])
    fig.savefig(OUT / "fig5_shap_summary.png", dpi=300)
    fig.savefig(OUT / "fig5_shap_summary.svg")
    plt.close(fig)


def partial_dependence_lines(model, x_ref: pd.DataFrame, feature: str, grid_size=45):
    values = x_ref[feature].to_numpy()
    lo, hi = np.quantile(values, [0.02, 0.98])
    grid = np.linspace(lo, hi, grid_size)
    yhat = []
    x_tmp = x_ref.copy()
    for val in grid:
        x_tmp[feature] = val
        yhat.append(model.predict(x_tmp).mean())
    return grid, np.asarray(yhat)


def plot_pdp(models, test_data):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=False)
    axes = axes.ravel()
    palette = sns.color_palette("tab10", n_colors=len(FEATURES))

    for ax, code in zip(axes, CATEGORY_FILES):
        x_test, _ = test_data[code]
        x_ref = x_test.sample(min(1500, len(x_test)), random_state=42).reset_index(drop=True)
        for color, feature in zip(palette, FEATURES):
            grid, yhat = partial_dependence_lines(models[code], x_ref, feature)
            ax.plot(grid, yhat, color=color, linewidth=1.8, label=FEATURE_LABELS[feature])

        ax.set_title(PANEL_TITLES[code], loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Perception score")
        ax.set_ylabel("Predicted satisfaction")
        ax.grid(color="#dddddd", linewidth=0.6)
        ax.legend(fontsize=8, ncol=2, frameon=False)

    fig.suptitle("Fig. 6 Dependency curves for different business types", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig6_dependency_curves.png", dpi=300)
    fig.savefig(OUT / "fig6_dependency_curves.svg")
    plt.close(fig)


def plot_interaction_heatmaps(models, shap_data):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
    axes = axes.ravel()
    matrices = {}

    for code in CATEGORY_FILES:
        x = shap_data[code]["x"].sample(
            min(120, len(shap_data[code]["x"])), random_state=42
        )
        explainer = shap.TreeExplainer(models[code])
        inter = explainer.shap_interaction_values(x)
        inter = np.asarray(inter)
        mat = np.abs(inter).mean(axis=0)
        np.fill_diagonal(mat, 0)
        matrices[code] = mat

    vmax = max(mat.max() for mat in matrices.values())
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    for ax, code in zip(axes, CATEGORY_FILES):
        sns.heatmap(
            matrices[code],
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            square=True,
            annot=True,
            fmt=".4f",
            xticklabels=labels,
            yticklabels=labels,
            cbar=code == "RTL",
            cbar_kws={"label": "Mean |SHAP interaction value|"},
        )
        ax.set_title(PANEL_TITLES[code], loc="left", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Fig. 7 Interaction contribution diagram of driving factors", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig7_interaction_contribution.png", dpi=300)
    fig.savefig(OUT / "fig7_interaction_contribution.svg")
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid", font="Arial")
    models, test_data, shap_data, metrics = fit_models()
    plot_shap_summary(shap_data)
    plot_pdp(models, test_data)
    plot_interaction_heatmaps(models, shap_data)
    print(metrics.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "R2": "{:.4f}".format,
    }))
    print(f"Saved outputs to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
