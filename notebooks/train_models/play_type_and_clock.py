import joblib
from sim_utils.modeling import create_model, create_reg_model, update_config
from utils.quack import Quack

print("import successful")


pbp = Quack.query("""
select 
    play_id,
    pbp.game_id,
    game_half,
    posteam,
    side_of_field,
    "desc",
    case when timeout =1 and posteam = timeout_team then 'pos_timeout'
         when timeout =1 and defteam = timeout_team then 'def_timeout'
    else play_type end as play_type,
    case when play_type = 'field_goal' then 1 else 0 end as is_field_goal,
    case when play_type = 'no_play' then 1 else 0 end as is_no_play,
    case when play_type = 'pass' then 1 else 0 end as is_pass,
    case when play_type = 'punt' then 1 else 0 end as is_punt,
    case when play_type = 'qb_kneel' then 1 else 0 end as is_qb_kneel,
    case when play_type = 'qb_spike' then 1 else 0 end as is_qb_spike,
    case when play_type = 'run' then 1 else 0 end as is_run,
    case when timeout = 1 then 1 else 0 end as is_timeout,
	COALESCE(
        quarter_seconds_remaining - LEAD(pbp.quarter_seconds_remaining) OVER (
            PARTITION BY pbp.game_id, pbp.qtr
            ORDER BY pbp.play_id
        ),
        pbp.quarter_seconds_remaining
	)::int as time_elapsed,
        quarter_seconds_remaining,
        half_seconds_remaining,
        game_seconds_remaining,
        yards_gained as yards,
        air_yards,
        incomplete_pass,
        out_of_bounds,
        timeout,
        sp,
        qtr,
    yardline_100,
    down,
    goal_to_go,
    ydstogo,
    posteam_score,
    score_differential,
    wind,
    temp,
    spread_line,
    total_line,
    posteam_timeouts_remaining,
    defteam_timeouts_remaining,
    qtr as quarter,
    o.*,
    t.*
    from pbp
    join team_feats as t
    on pbp.posteam = t.team
    and t.game_id = pbp.game_id
    join opp_feats as o
    on pbp.defteam = o.opponent_team
    and o.game_id = pbp.game_id
    where pbp.play_type <> 'no_play'
    order by pbp.game_id, play_id
                  """)
pbp['time_elapsed'] = pbp['time_elapsed'].clip(0, 60)

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
    "team",
    "away_team",
    "game_half",
    "posteam",
    "side_of_field",
    "desc",
]

play_type_mapping = {
    "field_goal": 0,
    "kickoff": 1,
    "pass": 2,
    "punt": 3,
    "qb_kneel": 4,
    "qb_spike": 5,
    "run": 6,
    "pos_timeout": 7,
    "def_timeout": 8,
    "no_play": 9,
}
pbp["play_type_enc"] = pbp["play_type"].map(play_type_mapping)

pbp["clock_stopped"] = 0
pbp.loc[
    (pbp.out_of_bounds == 1)
    | (pbp.timeout == 1)
    | (pbp.incomplete_pass == 1)
    | (pbp.sp == 1)
    | (pbp.play_type.isin(["no_play", None])),
    "clock_stopped",
] = 1

pbp["next_play"] = pbp.groupby("game_id")["play_type_enc"].shift(-1)

print("data preprocessed..")
clock_x = [
    "play_type_enc",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "yards",
    "incomplete_pass",
    "out_of_bounds",
    "is_field_goal",
    "is_no_play",
    "is_pass",
    "is_punt",
    "is_qb_kneel",
    "is_qb_spike",
    "is_run",
    "is_timeout"
]
clock_y = "time_elapsed"
print(pbp[clock_y].max(),pbp[clock_y].min())
clock_model = create_reg_model(pbp, clock_x, clock_y)

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
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "wind",
    "temp",
    "spread_line",
    "total_line",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "quarter",
] + team_stat_cols

y_col = ["play_type_enc"]
data = pbp.loc[~pbp.play_type_enc.isna(), x_cols + y_col + ["game_id"]].copy()
data.loc[data.play_type_enc.isin([7, 8]), "ydstogo"] = None
data[x_cols] = data.groupby("game_id")[x_cols].ffill(limit=1)
data = data.loc[~data.play_type_enc.isna()]
print(data['play_type_enc'].unique())
play_type_model = create_model(data, x_cols, y_col[0], colsample_bytree=0.8)
joblib.dump(play_type_model, "models/run_or_pass.joblib")

feature_config = {
    "run_or_pass_cols": x_cols,
    "play_decoding": {k: v for v, k in play_type_mapping.items()},
    "play_encoding": play_type_mapping
}
update_config(feature_config)
print("play type model saved.")
