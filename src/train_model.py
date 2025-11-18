import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

df = pd.read_csv("data/processed/games_with_features.csv")
df = df.dropna()

X = df[["home_pts_roll3", "away_pts_roll3", "home_pts_roll5", "away_pts_roll5"]]
y = df["total_points"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Baseline MAE: {mae:.2f} points")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/baseline_total_points_gb.pkl")
print("Model saved → models/baseline_total_points_gb.pkl")
