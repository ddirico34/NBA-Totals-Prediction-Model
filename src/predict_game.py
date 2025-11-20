from pathlib import Path
import argparse
import pandas as pd
import joblib


DATA_PATH = Path("data/processed/games_with_features.csv")
MODEL_PATH = Path("models/baseline_total_points_gb.pkl")


def load_data_and_model():
    df = pd.read_csv(DATA_PATH, parse_dates=["GAME_DATE"])
    model = joblib.load(MODEL_PATH)
    return df, model


def find_game_row(df, home_team, away_team, date_str=None, game_id=None):
    games = df[(df["home_team"] == home_team) & (df["away_team"] == away_team)]

    if game_id is not None:
        games = games[games["GAME_ID"] == game_id]

    if date_str is not None:
        target_date = pd.to_datetime(date_str).date()
        games = games[games["GAME_DATE"].dt.date == target_date]

    if games.empty:
        return None

    # If multiple games, pick the most recent one
    games = games.sort_values("GAME_DATE")
    return games.iloc[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Predict total points for a specific NBA game using the trained model."
    )
    parser.add_argument("home_team", help="Home team abbreviation (e.g. BOS)")
    parser.add_argument("away_team", help="Away team abbreviation (e.g. DAL)")
    parser.add_argument(
        "--date",
        help="Optional game date in YYYY-MM-DD (if omitted, uses most recent matchup)",
        default=None,
    )
    parser.add_argument(
        "--game-id",
        help="Optional specific GAME_ID if you know it",
        default=None,
    )

    args = parser.parse_args()

    df, model = load_data_and_model()

    row = find_game_row(
        df,
        home_team=args.home_team,
        away_team=args.away_team,
        date_str=args.date,
        game_id=args.game_id,
    )

    if row is None:
        print("❌ No matching game found for those inputs.")
        return

    # 🔑 Get the exact feature list the model was trained on
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        # Fallback (shouldn't happen with sklearn >= 1.0, but just in case)
        feature_cols = [
            "home_off_rating_simple",
            "away_off_rating_simple",
            "home_home_off_rating",
            "away_away_off_rating",
            "home_rest_days",
            "away_rest_days",
            "home_is_b2b",
            "away_is_b2b",
            "home_env_last5",
            "away_env_last5",
            "home_injury_impact",
            "away_injury_impact",
        ]

    # Make sure all needed columns exist for this row
    missing = [c for c in feature_cols if c not in row.index]
    if missing:
        print("❌ Missing feature columns in games_with_features.csv:")
        for m in missing:
            print("   -", m)
        return

    X = row[feature_cols].to_frame().T

    pred_total = float(model.predict(X)[0])
    actual_total = float(row["total_points"])
    game_date = row["GAME_DATE"].date()

    print(f"\n📅 Game date   : {game_date}")
    print(f"🏠 Home team   : {row['home_team']}")
    print(f"🛫 Away team   : {row['away_team']}")
    print(f"🎯 Pred total  : {pred_total:.2f}")
    print(f"📊 Actual total: {actual_total:.2f}")
    print(f"🔎 Error       : {abs(actual_total - pred_total):.2f} points\n")


if __name__ == "__main__":
    main()
