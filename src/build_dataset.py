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
    df = pd.read_csv(PROCESSED_DIR / "all_seasons_raw_team_games.csv", parse_dates=["GAME_DATE"])

    # Derive season_type from GAME_ID prefix (001=pre, 002=regular, 003=play-in, 004=playoffs)
    def get_season_type(game_id):
        s = str(game_id).zfill(10)  # pad just in case
        prefix = s[:3]
        if prefix == "001":
            return "Preseason"
        elif prefix == "002":
            return "Regular Season"
        elif prefix == "003":
            return "Play-In"
        elif prefix == "004":
            return "Playoffs"
        else:
            return "Unknown"

    df["season_type"] = df["GAME_ID"].apply(get_season_type)

    df["is_home"] = df["MATCHUP"].str.contains("vs.")
    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    # Merge home/away by GAME_ID + SEASON_ID
    merged = home.merge(
        away,
        on=["GAME_ID", "SEASON_ID"],
        suffixes=("_home", "_away"),
    )

    merged = merged[[
        "GAME_ID",
        "GAME_DATE_home",
        "SEASON_ID",
        "season_type_home",
        "TEAM_ABBREVIATION_home",
        "TEAM_ABBREVIATION_away",
        "PTS_home",
        "PTS_away",
    ]]

    merged = merged.rename(columns={
        "GAME_DATE_home": "GAME_DATE",
        "SEASON_ID": "season_id",
        "season_type_home": "season_type",
        "TEAM_ABBREVIATION_home": "home_team",
        "TEAM_ABBREVIATION_away": "away_team",
        "PTS_home": "home_points",
        "PTS_away": "away_points",
    })

    merged["total_points"] = merged["home_points"] + merged["away_points"]

    out_path = PROCESSED_DIR / "games_basic.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved basic game dataset → {out_path} with {len(merged)} games")


def add_team_ratings_with_rest_and_home_away():
    """
    For each game, add:
    - simple offensive/defensive ratings (overall)
    - home/away-specific offensive/defensive ratings
    - rest days + back-to-back flags
    - injury impact placeholders (0 for now)
    """
    df = pd.read_csv(PROCESSED_DIR / "games_basic.csv", parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    records = []

    # team -> stats dict
    team_stats = {}  # e.g. {"BOS": {...}}

    for _, row in df.iterrows():
        date = row["GAME_DATE"]
        home = row["home_team"]
        away = row["away_team"]
        hp = row["home_points"]
        ap = row["away_points"]

        def get_team_stats(team):
            return team_stats.get(
                team,
                {
                    "pts_for": 0,
                    "pts_against": 0,
                    "games": 0,
                    "home_pts_for": 0,
                    "home_pts_against": 0,
                    "home_games": 0,
                    "away_pts_for": 0,
                    "away_pts_against": 0,
                    "away_games": 0,
                    "last_game_date": None,
                },
            )

        # --- Get pre-game stats ---
        hs = get_team_stats(home)
        as_ = get_team_stats(away)

        def calc_overall_off_def(s):
            if s["games"] == 0:
                return 110.0, 110.0  # neutral prior
            return s["pts_for"] / s["games"], s["pts_against"] / s["games"]

        def calc_home_off_def(s):
            if s["home_games"] == 0:
                return calc_overall_off_def(s)
            return s["home_pts_for"] / s["home_games"], s["home_pts_against"] / s["home_games"]

        def calc_away_off_def(s):
            if s["away_games"] == 0:
                return calc_overall_off_def(s)
            return s["away_pts_for"] / s["away_games"], s["away_pts_against"] / s["away_games"]

        home_off, home_def = calc_overall_off_def(hs)
        away_off, away_def = calc_overall_off_def(as_)

        home_home_off, home_home_def = calc_home_off_def(hs)
        away_away_off, away_away_def = calc_away_off_def(as_)

        # --- Rest days & back-to-back flags ---
        def calc_rest_and_b2b(s):
            if s["last_game_date"] is None:
                # treat as well-rested
                return 5, 0
            diff = (date - s["last_game_date"]).days
            is_b2b = 1 if diff == 1 else 0
            return diff, is_b2b

        home_rest_days, home_is_b2b = calc_rest_and_b2b(hs)
        away_rest_days, away_is_b2b = calc_rest_and_b2b(as_)

        rec = row.to_dict()

        # overall ratings
        rec["home_off_rating_simple"] = home_off
        rec["home_def_rating_simple"] = home_def
        rec["away_off_rating_simple"] = away_off
        rec["away_def_rating_simple"] = away_def

        # home/away split ratings
        rec["home_home_off_rating"] = home_home_off
        rec["home_home_def_rating"] = home_home_def
        rec["away_away_off_rating"] = away_away_off
        rec["away_away_def_rating"] = away_away_def

        # rest & b2b
        rec["home_rest_days"] = home_rest_days
        rec["away_rest_days"] = away_rest_days
        rec["home_is_b2b"] = home_is_b2b
        rec["away_is_b2b"] = away_is_b2b

        # injury placeholders (0 for now – later you can fill these in from another source)
        rec["home_injury_impact"] = 0.0
        rec["away_injury_impact"] = 0.0

        records.append(rec)

        # --- update stats AFTER the game ---
        # home team
        hs["pts_for"] += hp
        hs["pts_against"] += ap
        hs["games"] += 1
        hs["home_pts_for"] += hp
        hs["home_pts_against"] += ap
        hs["home_games"] += 1
        hs["last_game_date"] = date
        team_stats[home] = hs

        # away team
        as_["pts_for"] += ap
        as_["pts_against"] += hp
        as_["games"] += 1
        as_["away_pts_for"] += ap
        as_["away_pts_against"] += hp
        as_["away_games"] += 1
        as_["last_game_date"] = date
        team_stats[away] = as_

    feat_df = pd.DataFrame(records)
    out_path = PROCESSED_DIR / "games_with_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"Saved game features (with rest/home-away/injury placeholders) to {out_path}")

def merge_injury_impact():
    games_path = PROCESSED_DIR / "games_with_features.csv"
    df = pd.read_csv(games_path)

    inj_path = PROCESSED_DIR / "injury_impact_by_game.csv"
    if not inj_path.exists():
        print("No injury impact file found, keeping existing injury_impact columns.")
        return

    inj = pd.read_csv(inj_path)

    # Merge on GAME_ID; if missing, treat as 0 impact
    df = df.merge(inj, on="GAME_ID", how="left", suffixes=("", "_inj"))

    for col in ["home_injury_impact", "away_injury_impact"]:
        inj_col = col + "_inj"
        if inj_col in df.columns:
            # overwrite with injury CSV where available; else keep old / 0
            df[col] = df[inj_col].fillna(df.get(col, 0.0))
            df.drop(columns=[inj_col], inplace=True)
        else:
            # if column didn't exist before, create it from CSV or 0
            df[col] = df[col].fillna(0.0) if col in df.columns else 0.0

    df.to_csv(games_path, index=False)
    print(f"Merged injury impact into {games_path}")

def merge_injury_impact():
    games_path = PROCESSED_DIR / "games_with_features.csv"
    df = pd.read_csv(games_path)

    inj_path = PROCESSED_DIR / "injury_impact_by_game.csv"
    if not inj_path.exists():
        print("No injury impact file found, skipping injury merge.")
        return

    # Try to read the injury file; handle empty file gracefully
    try:
        inj = pd.read_csv(inj_path)
    except pd.errors.EmptyDataError:
        print("injury_impact_by_game.csv is empty, skipping injury merge.")
        return

    if inj.empty:
        print("injury_impact_by_game.csv has no rows, skipping injury merge.")
        return

    # We expect inj to have: GAME_ID, home_injury_impact, away_injury_impact
    if "GAME_ID" not in inj.columns:
        print("injury_impact_by_game.csv missing GAME_ID column, skipping injury merge.")
        return

    # Merge on GAME_ID
    df = df.merge(inj, on="GAME_ID", how="left", suffixes=("", "_inj"))

    # Make sure these columns exist, even if they didn't before
    if "home_injury_impact" not in df.columns:
        df["home_injury_impact"] = 0.0
    if "away_injury_impact" not in df.columns:
        df["away_injury_impact"] = 0.0

    # If there were *_inj columns, use them to overwrite base cols
    if "home_injury_impact_inj" in df.columns:
        df["home_injury_impact"] = df["home_injury_impact_inj"].fillna(df["home_injury_impact"])
        df.drop(columns=["home_injury_impact_inj"], inplace=True)

    if "away_injury_impact_inj" in df.columns:
        df["away_injury_impact"] = df["away_injury_impact_inj"].fillna(df["away_injury_impact"])
        df.drop(columns=["away_injury_impact_inj"], inplace=True)

    # Finally, fill any remaining NaNs with 0
    df["home_injury_impact"] = df["home_injury_impact"].fillna(0.0)
    df["away_injury_impact"] = df["away_injury_impact"].fillna(0.0)

    df.to_csv(games_path, index=False)
    print(f"Merged injury impact into {games_path}")



if __name__ == "__main__":
    combine_seasons()
    build_single_row_games()
    add_team_ratings_with_rest_and_home_away()
    merge_injury_impact()

