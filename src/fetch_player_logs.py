import time
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

def fetch_season_logs(season: str):
    """
    Fetch all player game logs for a given season.
    Older nba_api versions do NOT use player_or_team argument.
    They return player logs by default.
    """
    print(f"\n=== Fetching player logs for {season} ===")

    log = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    )

    df = log.get_data_frames()[0]

    out_path = RAW_DIR / f"player_logs_{season}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path} ({len(df)} rows)")

def main():
    for season in SEASONS:
        fetch_season_logs(season)
        time.sleep(1)

if __name__ == "__main__":
    main()
