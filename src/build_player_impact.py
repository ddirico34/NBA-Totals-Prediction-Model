from pathlib import Path
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import PlayerGameLog

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Seasons in the format that PlayerGameLog expects
SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

# Focus on true impact stars first (you can add to this list anytime)
STAR_PLAYERS = [
    "Joel Embiid",
    "Luka Doncic",
    "Nikola Jokic",
    "Stephen Curry",
    "Giannis Antetokounmpo",
    "Jayson Tatum",
    "Shai Gilgeous-Alexander",
    "LeBron James",
    "Damian Lillard",
    "Devin Booker",
]


def find_player_id_by_name(name: str):
    matches = players.find_players_by_full_name(name)
    if not matches:
        print(f"Warning: no player_id found for {name}")
        return None
    return matches[0]["id"]


def fetch_star_logs() -> pd.DataFrame:
    """Fetch game logs for a small set of star players across multiple seasons."""
    all_rows = []

    for name in STAR_PLAYERS:
        player_id = find_player_id_by_name(name)
        if player_id is None:
            continue

        for season in SEASONS:
            try:
                print(f"Fetching logs for {name} in {season} (id={player_id})...")
                log = PlayerGameLog(player_id=player_id, season=season)
                df = log.get_data_frames()[0]

                if df.empty:
                    continue

                # Add metadata columns we want to keep
                df["PLAYER_NAME"] = name
                df["PLAYER_ID"] = player_id
                df["season_id"] = season

                all_rows.append(df)

                time.sleep(0.6)  # be nice to the API

            except Exception as e:
                print(f"Error fetching logs for {name} {season}: {e}")
                time.sleep(1.0)
                continue

    if not all_rows:
        raise RuntimeError("No logs fetched for any star players.")
    return pd.concat(all_rows, ignore_index=True)


def compute_player_impact():
    df = fetch_star_logs()

    # Ensure needed numeric columns exist and are numeric
    numeric_cols = ["PTS", "FGA", "FTA", "TOV", "PLUS_MINUS", "MIN"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Filter out games with 0 minutes (DNP)
    df = df[df["MIN"] > 0].copy()

    # Simple usage proxy: FGA + 0.44*FTA + TOV
    df["usage_possessions"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]

    # NOTE: your Game ID column is named "Game_ID", not "GAME_ID"
    group_cols = ["PLAYER_ID", "PLAYER_NAME", "season_id"]

    grp = df.groupby(group_cols).agg(
        games_played=("Game_ID", "nunique"),
        total_minutes=("MIN", "sum"),
        total_points=("PTS", "sum"),
        total_usage_possessions=("usage_possessions", "sum"),
        avg_plus_minus=("PLUS_MINUS", "mean"),
    ).reset_index()

    # Per-game stats
    grp["ppg"] = grp["total_points"] / grp["games_played"].clip(lower=1)
    grp["mpg"] = grp["total_minutes"] / grp["games_played"].clip(lower=1)
    grp["usage_per_game"] = grp["total_usage_possessions"] / grp["games_played"].clip(lower=1)

    # Normalize usage to z-score so it doesn't dwarf PPG
    usage_mean = grp["usage_per_game"].mean()
    usage_std = grp["usage_per_game"].std(ddof=0) or 1.0
    grp["usage_z"] = (grp["usage_per_game"] - usage_mean) / usage_std

    # Impact score: PPG + (0.5 * plus_minus) + (2 * usage_z)
    w_ppg = 1.0
    w_plus = 0.5
    w_usage = 2.0

    grp["impact_score"] = (
        w_ppg * grp["ppg"]
        + w_plus * grp["avg_plus_minus"]
        + w_usage * grp["usage_z"]
    )

    # Filter to players with enough games to be meaningful
    grp = grp[grp["games_played"] >= 10].copy()

    out_path = PROCESSED_DIR / "player_impact_scores.csv"
    grp.to_csv(out_path, index=False)
    print(f"Saved player impact scores → {out_path}")


if __name__ == "__main__":
    compute_player_impact()
