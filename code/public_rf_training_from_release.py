from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rf_model_input_anonymized.csv"
OUT = BASE / "outputs" / "reproduced_model_results"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["Aesthetics", "Safety", "Vibrancy", "Wealth", "Depression", "Boring"]
CATEGORIES = ["CAT", "ISS", "LEX", "RTL"]

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


def main():
    df = pd.read_csv(DATA)
    metric_rows = []
    importance_rows = []
    for category in CATEGORIES:
        data = df[df["category"] == category].dropna(subset=["satisfaction_score"] + FEATURES)
        x = data[FEATURES]
        y = data["satisfaction_score"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        metric_rows.append(
            {
                "category": category,
                "rows_used": len(data),
                "train_n": len(x_train),
                "test_n": len(x_test),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred, squared=False),
                "R2": r2_score(y_test, pred),
            }
        )
        for feature, value in zip(FEATURES, model.feature_importances_):
            importance_rows.append({"category": category, "feature": feature, "rf_importance": value})

    pd.DataFrame(metric_rows).to_csv(OUT / "rf_metrics_reproduced.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(OUT / "rf_feature_importance_reproduced.csv", index=False)
    print(f"Saved reproduced results to {OUT}")


if __name__ == "__main__":
    main()
