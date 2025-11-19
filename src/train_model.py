from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = Path("data/processed/games_with_features.csv")
MODELS_DIR = Path("models")


def train_model():
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # ✅ Only keep regular season games
    df = df[df["season_type"] == "Regular Season"].copy()


    # drop any rows with NaNs from early games
    df = df.dropna(subset=[
    "home_off_rating_simple", "away_off_rating_simple",
    "home_home_off_rating", "away_away_off_rating",
    "home_rest_days", "away_rest_days",
    "home_is_b2b", "away_is_b2b",
    "home_injury_impact", "away_injury_impact",
])


    split_idx = int(0.8 * len(df))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    feature_cols = [
    "home_off_rating_simple",
    "away_off_rating_simple",
    "home_home_off_rating",
    "away_away_off_rating",
    "home_rest_days",
    "away_rest_days",
    "home_is_b2b",
    "away_is_b2b",
    "home_injury_impact",
    "away_injury_impact",
]


    X_train = train[feature_cols]
    y_train = train["total_points"]

    X_test = test[feature_cols]
    y_test = test["total_points"]

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"Train MAE: {mae_train:.2f} points")
    print(f"Test  MAE: {mae_test:.2f} points")

    test = test.copy()
    test["pred_total"] = y_test_pred

    print("\nSample test games (actual vs predicted totals):")
    print(
        test[["GAME_DATE", "home_team", "away_team", "total_points", "pred_total"]]
        .tail(5)
        .to_string(index=False)
    )

    model_path = MODELS_DIR / "baseline_total_points_gb.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    train_model()
