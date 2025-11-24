from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_player_impacts():
    path = PROCESSED_DIR / "player_impact_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            "player_impact_scores.csv not found. "
            "Run `py -m src.build_player_impact` first."
        )
    df = pd.read_csv(path)
    # Normalize name for matching
    df["PLAYER_NAME_norm"] = df["PLAYER_NAME"].str.lower().str.strip()
    return df


def parse_player_list(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [p.strip() for p in str(s).split(";") if p.strip()]


def build_injury_impact():
    events_path = PROCESSED_DIR / "injury_events.csv"
    if not events_path.exists():
        raise FileNotFoundError(
            "injury_events.csv not found. "
            "Create it in data/processed with GAME_ID, home_out_players, away_out_players."
        )

    events = pd.read_csv(events_path)
    player_impacts = load_player_impacts()

    rows = []

    def sum_impact(names_list):
        total = 0.0
        for name in names_list:
            name_norm = name.lower().strip()
            candidates = player_impacts[
                player_impacts["PLAYER_NAME_norm"] == name_norm
            ]
            if candidates.empty:
                print(f"Warning: no impact score for {name}")
                continue
            impact_val = candidates["impact_score"].max()
            total += impact_val
        return total

    for _, ev in events.iterrows():
        game_id = ev["GAME_ID"]
        home_out = parse_player_list(ev.get("home_out_players", ""))
        away_out = parse_player_list(ev.get("away_out_players", ""))

        rows.append(
            {
                "GAME_ID": game_id,
                "home_injury_impact": sum_impact(home_out),
                "away_injury_impact": sum_impact(away_out),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "injury_impact_by_game.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved injury impact per game → {out_path}")


if __name__ == "__main__":
    build_injury_impact()
