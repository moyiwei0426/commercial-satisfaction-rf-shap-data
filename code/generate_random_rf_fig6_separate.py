from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

CATEGORY_NAMES = {
    "CAT": "Catering Services",
    "ISS": "In-store Service Sectors",
    "LEX": "Leisure and Experience Venues",
    "RTL": "Retail and Trade Locations",
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


def read_model_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gb18030", low_memory=False)
    model_df = df[["star"] + FEATURES].copy()
    for col in ["star"] + FEATURES:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna(subset=["star"] + FEATURES)


def partial_dependence_line(model, x_ref: pd.DataFrame, feature: str, grid_size=45):
    values = x_ref[feature].to_numpy()
    lo, hi = np.quantile(values, [0.02, 0.98])
    grid = np.linspace(lo, hi, grid_size)
    yhat = []
    for val in grid:
        x_tmp = x_ref.copy()
        x_tmp[feature] = val
        yhat.append(model.predict(x_tmp).mean())
    return grid, np.asarray(yhat)


def plot_category_pdp(code: str, model, x_test: pd.DataFrame):
    x_ref = x_test.sample(min(600, len(x_test)), random_state=42).reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    axes = axes.ravel()
    color = "#2f6fbb"

    for ax, feature in zip(axes, FEATURES):
        grid, yhat = partial_dependence_line(model, x_ref, feature)
        ax.plot(grid, yhat, color=color, linewidth=2.0)
        ax.scatter(grid, yhat, color=color, s=11, alpha=0.55)
        ax.set_title(FEATURE_LABELS[feature], loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Perception score")
        ax.set_ylabel("Predicted satisfaction")
        ax.grid(color="#dddddd", linewidth=0.6)

    fig.suptitle(
        f"Fig. 6 Dependency curves - {CATEGORY_NAMES[code]} ({code})",
        y=0.995,
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"fig6_{code.lower()}_dependency_panels.png", dpi=300)
    fig.savefig(OUT / f"fig6_{code.lower()}_dependency_panels.svg")
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid", font="Arial")
    for code, path in CATEGORY_FILES.items():
        print(f"Training and plotting {code}...", flush=True)
        data = read_model_data(path)
        x = data[FEATURES]
        y = data["star"]
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.25, random_state=42
        )
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(x_train, y_train)
        plot_category_pdp(code, model, x_test.reset_index(drop=True))

    print(f"Saved outputs to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
