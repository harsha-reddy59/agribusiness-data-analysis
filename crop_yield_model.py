"""
Week 4 - Predictive Modeling Framework for Agricultural Yield Forecasting
-------------------------------------------------------------------------
Reusable pipeline: data -> feature engineering -> time-aware split ->
model comparison -> cross-validation -> evaluation -> feature importance.

To apply to your own data, replace `load_data()` with a CSV/DB loader and
update TARGET, DATE_COL, NUM_FEATURES and CAT_FEATURES. The rest is unchanged.

Requirements: pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TARGET = "yield_t_ha"
DATE_COL = "year"
NUM_FEATURES = ["rainfall_mm", "avg_temp_c", "gdd", "soil_ph", "soil_n_ppm",
                "fertilizer_kg_ha", "irrigation_mm", "ndvi_peak", "rain_x_temp",
                "prev_yield"]
CAT_FEATURES = ["crop", "region", "soil_type"]


# ---------------------------------------------------------------- 1. DATA
def load_data(n=1500):
    """Synthetic dataset so the script runs end-to-end.
    Replace with: return pd.read_csv('your_data.csv')"""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "year": rng.integers(2005, 2025, n),
        "crop": rng.choice(["wheat", "rice", "maize"], n),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "soil_type": rng.choice(["loam", "clay", "sandy"], n),
        "rainfall_mm": rng.normal(750, 150, n),
        "avg_temp_c": rng.normal(24, 3, n),
        "soil_ph": rng.normal(6.6, 0.6, n),
        "soil_n_ppm": rng.normal(40, 10, n),
        "fertilizer_kg_ha": rng.normal(120, 30, n),
        "irrigation_mm": rng.normal(200, 60, n),
        "ndvi_peak": rng.normal(0.72, 0.08, n),
    })
    df["gdd"] = df["avg_temp_c"] * 180 - 1500 + rng.normal(0, 60, n)
    df["prev_yield"] = rng.normal(4.2, 0.8, n)
    df[TARGET] = (
        1.2 + 0.0022 * df.rainfall_mm - 0.045 * (df.avg_temp_c - 24) ** 2
        + 0.012 * df.fertilizer_kg_ha + 0.004 * df.irrigation_mm
        + 3.2 * df.ndvi_peak - 0.25 * np.abs(df.soil_ph - 6.5)
        + 0.25 * df.prev_yield + rng.normal(0, 0.35, n)
    )
    # inject missing values to demonstrate imputation
    for c in ["soil_n_ppm", "soil_ph", "irrigation_mm"]:
        df.loc[rng.choice(n, int(0.05 * n), replace=False), c] = np.nan
    return df


# ------------------------------------------------- 2. FEATURE ENGINEERING
def engineer(df):
    df = df.copy()
    df["rain_x_temp"] = df["rainfall_mm"] * df["avg_temp_c"] / 100  # interaction
    return df


# ---------------------------------------------------------- 3. PREPROCESS
def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM_FEATURES),
                              ("cat", cat, CAT_FEATURES)])


# --------------------------------------------------------------- 4. MODELS
def candidate_models():
    return {
        "Ridge (baseline)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)}


# -------------------------------------------------------------------- MAIN
def main():
    df = engineer(load_data()).sort_values(DATE_COL).reset_index(drop=True)

    # Time-aware split: train on the past, test on the most recent years
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_tr, y_tr = train[NUM_FEATURES + CAT_FEATURES], train[TARGET]
    X_te, y_te = test[NUM_FEATURES + CAT_FEATURES], test[TARGET]
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")

    # Baseline: always predict the training mean
    base = metrics(y_te, np.full(len(y_te), y_tr.mean()))
    print("\nNaive baseline:", {k: round(v, 3) for k, v in base.items()})

    # Cross-validation (forward-chaining) on the training set
    tscv = TimeSeriesSplit(n_splits=5)
    results, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=tscv,
                             scoring="neg_root_mean_squared_error")
        pipe.fit(X_tr, y_tr)
        m = metrics(y_te, pipe.predict(X_te))
        results.append({"Model": name, "CV_RMSE": -cv.mean(),
                        "CV_STD": cv.std(), **m})
        fitted[name] = pipe
    table = pd.DataFrame(results).set_index("Model").round(3)
    print("\nModel comparison:\n", table)

    # Hyper-parameter tuning of the strongest tree model
    grid = GridSearchCV(
        Pipeline([("prep", build_preprocessor()),
                  ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
        {"model__n_estimators": [200, 400],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
        cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    pred = best.predict(X_te)
    final = metrics(y_te, pred)
    print("\nBest params:", grid.best_params_)
    print("Tuned model on hold-out:", {k: round(v, 3) for k, v in final.items()})

    # Uncertainty: simple residual-based 90% prediction interval
    resid = y_tr - best.predict(X_tr)
    lo, hi = np.percentile(resid, [5, 95])
    cover = np.mean((y_te >= pred + lo) & (y_te <= pred + hi))
    print(f"90% interval coverage on test set: {cover:.1%}")

    # Feature importance (permutation based, model-agnostic)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best, X_te, y_te, n_repeats=10,
                                random_state=RANDOM_STATE)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    print("\nTop drivers of yield:\n", imp.sort_values(ascending=False).head(6).round(3))

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(y_te, pred, s=10, alpha=.6)
    lims = [y_te.min(), y_te.max()]
    ax[0].plot(lims, lims, "r--")
    ax[0].set(xlabel="Actual yield (t/ha)", ylabel="Predicted yield (t/ha)",
              title="Actual vs Predicted")
    imp.plot.barh(ax=ax[1], title="Permutation importance")
    plt.tight_layout()
    plt.savefig("model_results.png", dpi=150)
    print("\nSaved model_results.png")


if __name__ == "__main__":
    main()
