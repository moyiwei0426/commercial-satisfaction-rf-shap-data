from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
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
PANEL_TITLES = {
    "CAT": "(a) CAT",
    "ISS": "(b) ISS",
    "LEX": "(c) LEX",
    "RTL": "(d) RTL",
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


def read_model_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gb18030", low_memory=False)
    model_df = df[["star"] + FEATURES].copy()
    for col in ["star"] + FEATURES:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna(subset=["star"] + FEATURES)


def train_models():
    models = {}
    test_data = {}
    for code, path in CATEGORY_FILES.items():
        print(f"Training {code}...", flush=True)
        data = read_model_data(path)
        x = data[FEATURES]
        y = data["star"]
        x_train, x_test, _, _ = train_test_split(
            x, y, test_size=0.25, random_state=42
        )
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(x_train, y.loc[x_train.index])
        models[code] = model
        test_data[code] = x_test.reset_index(drop=True)
    return models, test_data


def partial_dependence_lines(model, x_ref: pd.DataFrame, feature: str, grid_size=25):
    values = x_ref[feature].to_numpy()
    lo, hi = np.quantile(values, [0.02, 0.98])
    grid = np.linspace(lo, hi, grid_size)
    yhat = []
    for val in grid:
        x_tmp = x_ref.copy()
        x_tmp[feature] = val
        yhat.append(model.predict(x_tmp).mean())
    return grid, np.asarray(yhat)


def plot_pdp(models, test_data):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=False)
    axes = axes.ravel()
    palette = sns.color_palette("tab10", n_colors=len(FEATURES))

    for ax, code in zip(axes, CATEGORY_FILES):
        print(f"Plotting PDP {code}...", flush=True)
        x_ref = test_data[code].sample(
            min(300, len(test_data[code])), random_state=42
        ).reset_index(drop=True)
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


def plot_interaction_heatmaps(models, test_data):
    matrices = {}
    for code in CATEGORY_FILES:
        print(f"Computing SHAP interaction {code}...", flush=True)
        x = test_data[code].sample(min(40, len(test_data[code])), random_state=42)
        explainer = shap.TreeExplainer(models[code])
        inter = np.asarray(explainer.shap_interaction_values(x))
        mat = np.abs(inter).mean(axis=0)
        np.fill_diagonal(mat, 0)
        matrices[code] = mat

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
    axes = axes.ravel()
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
    models, test_data = train_models()
    plot_pdp(models, test_data)
    plot_interaction_heatmaps(models, test_data)
    print(f"Saved outputs to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
