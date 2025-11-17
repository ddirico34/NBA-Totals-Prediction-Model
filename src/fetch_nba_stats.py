from nba_api.stats.endpoints import LeagueGameFinder
import pandas as pd
from pathlib import Path


def fetch_games_for_season(season="2023-24", save_dir="data/raw"):
    print(f"Fetching NBA games for season {season}...")

    # Fetch data
    gamefinder = LeagueGameFinder(season_nullable=season)
    df = gamefinder.get_data_frames()[0]

    # Keep useful columns
    cols = [
        "GAME_ID", "GAME_DATE", "SEASON_ID",
        "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL", "PTS"
    ]
    df = df[cols]

    # Create output directory if needed
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    out_file = save_path / f"games_{season.replace('-', '_')}.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24"]
    for season in seasons:
        fetch_games_for_season(season)

