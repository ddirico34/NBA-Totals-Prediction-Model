# NBA Over/Under Prediction Project

I'm building a Python-based model to predict NBA game totals and eventually aim to achieve at least 60% accuracy on Over/Under picks using real data and machine learning. This is an ongoing project that I will update daily to track progress and show consistent development.

### Progress Log

- **Day 1** – Project setup, environment fixed, and raw NBA game data pulled with `nba_api` starting from 2020-2024
- **Day 2** – Combined multi-season data, built a one-row-per-game dataset, and trained a first baseline model (MAE ≈ 13.6 points).
- **Day 3** - Added team-based ratings (home/away scoring, rest days, back-to-back flags), filtered evaluation to regular-season games only, wired in the injury impact pipeline, and retrained the model
- **Day 4** - Added a prediction script (predict_game.py) that auto-detects model features, introduced a pace-based environment feature, and rebuilt the dataset. Model performance improved again, lowering Train MAE from 13.6 (Day 2) to 12.85. The model can now run accurate historical game predictions directly from the command line.

### Tools (so far)
- Python
- `nba_api`
- `pandas`

### Goal
Predict total points per game, compare to sportsbook line, and output **Over**, **Under**, or **No Bet**.
