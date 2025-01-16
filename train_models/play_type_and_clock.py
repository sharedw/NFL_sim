import pandas as pd
import yaml
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sim_utils.modeling import create_model, create_reg_model, update_config



pbp = pd.read_parquet("data/pbp.parquet")

team = pd.read_csv("data/agg/team_stats.csv", index_col=0)
opp = pd.read_csv("data/agg/opp_stats.csv", index_col=0)
team_stat_cols = [
    "completions_team_roll",
    "attempts_team_roll",
    "carries_team_roll",
    "passing_yards_team_roll",
    "rushing_yards_team_roll",
    "pass_pct_team_roll",
    "completions_opp_roll",
    "attempts_opp_roll",
    "carries_opp_roll",
    "passing_yards_opp_roll",
    "rushing_yards_opp_roll",
    "pass_pct_opp_roll",
]
print("data read in successfully..")
context_cols = [
    "play_id",
    "game_id",
    "home_team",
    "away_team",
    "game_half",
    "posteam",
    "side_of_field",
    "desc",
]


pbp = pbp.merge(
    team,
    left_on=["posteam", "season", "week"],
    right_on=["recent_team", "season", "week"],
).drop("recent_team", axis=1)
pbp = pbp.merge(
    opp,
    left_on=["defteam", "season", "week"],
    right_on=["opponent_team", "season", "week"],
).drop("opponent_team", axis=1)

pbp.loc[(pbp.timeout == 1) & (pbp.posteam == pbp.timeout_team), "play_type"] = (
    "pos_timeout"
)
pbp.loc[(pbp.timeout == 1) & (pbp.defteam == pbp.timeout_team), "play_type"] = (
    "def_timeout"
)

play_type_mapping = {
    "field_goal": 0,
    "no_play": 1,
    "pass": 2,
    "punt": 3,
    "qb_kneel": 4,
    "qb_spike": 5,
    "run": 6,
    "pos_timeout": 7,
    "def_timeout": 8,
    "kickoff": 9,
}
pbp["play_type_enc"] = pbp["play_type"].map(play_type_mapping)

df = pbp[
    [
        "game_id",
        "desc",
        "play_type",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "yards_gained",
        "air_yards",
        "incomplete_pass",
        "out_of_bounds",
        "timeout",
        "sp",
        "play_type_enc",
    ]
].copy()
df["time_elapsed"] = df["half_seconds_remaining"] - df.groupby("game_id")[
    "half_seconds_remaining"
].shift(-1)
df["clock_stopped"] = 0
df.loc[
    (df.out_of_bounds == 1)
    | (df.timeout == 1)
    | (df.incomplete_pass == 1)
    | (df.sp == 1)
    | (df.play_type.isin(["no_play", None])),
    "clock_stopped",
] = 1

df["next_play"] = df.groupby("game_id")["play_type_enc"].shift(-1)

print("data preprocessed..")


clock_x = [
    "play_type_enc",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "yards_gained",
    "incomplete_pass",
    "out_of_bounds",
    "timeout",
    "sp",
    "next_play",
]
clock_y = "time_elapsed"
clock_model = create_reg_model(df, clock_x, clock_y)

print("clock model created..")
joblib.dump(clock_model, "models/clock_model.joblib")

feature_config = {
    "clock_cols": clock_x,
}
update_config(feature_config)
print("clock model saved.")

x_cols = [
    "yardline_100",
    "down",
    "goal_to_go",
    "ydstogo",
    "posteam_score",
    "score_differential",
    'quarter_seconds_remaining',
    'half_seconds_remaining',
    'game_seconds_remaining',
    "wind",
    "temp",
    'spread_line',
    'total_line',
    'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    'quarter',
] + team_stat_cols

pbp['quarter'] = pbp['qtr']
y_col = ["play_type_enc"]
data = pbp.loc[~pbp.play_type_enc.isna(), x_cols + y_col+['game_id']].copy()
data.loc[data.play_type_enc.isin([7,8]),'ydstogo'] = None
data[x_cols] = data.groupby('game_id')[x_cols].ffill(limit=1)
data = data.loc[~data.play_type_enc.isna()]

play_type_model = create_model(data, x_cols, y_col[0], colsample_bytree=0.8)
joblib.dump(play_type_model, 'models/run_or_pass.joblib')

feature_config = {
    'run_or_pass_cols':x_cols,
    'play_encoding': {k:v for v,k in play_type_mapping.items()}
}
update_config(feature_config)
print('play type model saved.')