from typing import TypedDict

class GameStateDict(TypedDict):
    possession: str
    quarter: int
    down: int
    is_first_down: bool
    ydstogo: int
    goal_to_go: int
    yardline_100: int
    total_home_score: int
    total_away_score: int
    posteam_score: int
    defteam_score: int
    score_differential: int
    wind: int
    temp: int
    quarter_seconds_remaining: int
    half_seconds_remaining: int
    game_seconds_remaining: int
    drive: int
    spread_line: int
    total_line: int
    posteam_timeouts_remaining: int
    defteam_timeouts_remaining: int