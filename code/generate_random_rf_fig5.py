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


def main():
    sns.set_theme(style="whitegrid", font="Arial")
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035], wspace=0.28, hspace=0.25)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])
    rng = np.random.default_rng(42)

    for ax, code in zip(axes, CATEGORY_FILES):
        print(f"Training and explaining {code}...", flush=True)
        data = read_model_data(CATEGORY_FILES[code])
        x = data[FEATURES]
        y = data["star"]
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.25, random_state=42
        )
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(x_train, y_train)
        sample_x = x_test.sample(min(350, len(x_test)), random_state=42).reset_index(drop=True)
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(sample_x, check_additivity=False))
        order = np.argsort(np.abs(values).mean(axis=0))[::-1]

        for y_pos, feature_idx in enumerate(order):
            feat_v = sample_x.iloc[:, feature_idx].to_numpy()
            shap_v = values[:, feature_idx]
            denom = np.nanmax(feat_v) - np.nanmin(feat_v)
            norm_v = (feat_v - np.nanmin(feat_v)) / denom if denom else np.zeros_like(feat_v)
            jitter = rng.normal(0, 0.075, len(shap_v))
            ax.scatter(
                shap_v,
                np.full(len(shap_v), y_pos) + jitter,
                c=norm_v,
                cmap="coolwarm",
                s=9,
                alpha=0.68,
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
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Feature value (low to high)")
    fig.suptitle("Fig. 5 Summary chart of feature importance SHAP values", y=0.99)
    fig.savefig(OUT / "fig5_shap_summary_v2.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig5_shap_summary_v2.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved outputs to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
