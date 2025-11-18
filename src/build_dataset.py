import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def combine_seasons():
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = PROCESSED_DIR / "all_seasons_raw_team_games.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"Saved combined raw dataset → {out_path}")

def build_single_row_games():
    df = pd.read_csv(PROCESSED_DIR / "all_seasons_raw_team_games.csv")

    df["is_home"] = df["MATCHUP"].str.contains("vs.")
    home = df[df["is_home"]]
    away = df[~df["is_home"]]

    merged = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_home", "_away")
    )

    merged = merged[[
        "GAME_ID",
        "GAME_DATE_home",
        "TEAM_ABBREVIATION_home",
        "TEAM_ABBREVIATION_away",
        "PTS_home",
        "PTS_away"
    ]]

    merged = merged.rename(columns={
        "GAME_DATE_home": "GAME_DATE",
        "TEAM_ABBREVIATION_home": "home_team",
        "TEAM_ABBREVIATION_away": "away_team",
        "PTS_home": "home_points",
        "PTS_away": "away_points",
    })

    merged["total_points"] = merged["home_points"] + merged["away_points"]

    out_path = PROCESSED_DIR / "games_basic.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved basic game dataset → {out_path}")

def add_features():
    df = pd.read_csv(PROCESSED_DIR / "games_basic.csv")

    df["home_pts_roll3"] = df["home_points"].rolling(3).mean()
    df["away_pts_roll3"] = df["away_points"].rolling(3).mean()

    df["home_pts_roll5"] = df["home_points"].rolling(5).mean()
    df["away_pts_roll5"] = df["away_points"].rolling(5).mean()

    out_path = PROCESSED_DIR / "games_with_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset with features → {out_path}")

if __name__ == "__main__":
    combine_seasons()
    build_single_row_games()
    add_features()
