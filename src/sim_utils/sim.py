from sim_utils.game import GameState
from sim_utils.team import Team
from utils.quack import Quack
import pandas as pd

def sim_games(
    team1_name: str, team2_name: str, season: int, week: int, n_games: int = 100
) -> pd.DataFrame:
    result = []

    t1 = Team(team1_name, season, week)
    t2 = Team(team2_name, season, week)

    game = GameState(t1, t2)

    for _ in range(n_games):
        game.start_game()
        game.play_game()
        result += game.game_results()
        game.reset_game()

    df = pd.DataFrame(result)[
        [
            "name",
            "team",
            "carries",
            "receptions",
            "targets",
            "rushing_yards",
            "receiving_yards",
            "passing_yards"
        ]
    ]
    df = df[df.ne(0).sum(axis=1) > 2]  # print only players with stats
    return df


def compare_stats(
    sim_result, stats, season=2024, week=0, min_filter=None
):
    select_cols = ", ".join([f"avg({s}) as {s}" for s in stats])
    sql = f"""
        select player_display_name as name, team, {select_cols}
        from weekly
        where season = {season} and week = {week}
        group by all
    """
    real = Quack.query(sql)
    print(real)
    med = sim_result.groupby(["name", "team"])[stats].median().round(2).reset_index()

    comb = real.merge(med, on=["name", "team"], suffixes=("_real", "_pred"))
    
    ordered_cols = ["name", "team"]
    for s in stats:
        ordered_cols += [f"{s}_real", f"{s}_pred"]
    comb = comb[ordered_cols]
    print(comb[comb.ne(0).sum(axis=1) > 2])
    
    results = {}
    for s in stats:
        res_col = f"{s}_res"
        comb[res_col] = comb[f"{s}_real"] > comb[f"{s}_pred"]
        tmp = comb
        if min_filter and s in min_filter:
            tmp = tmp[tmp[f"{s}_pred"] > min_filter[s]]
        results[s] = tmp[res_col].mean()
    return results


week = 8
season = 2025

result = sim_games("KC", "WAS", season, week, n_games=20)
comp = compare_stats(
    result,
    ["receptions", "carries", "receiving_yards", "rushing_yards", "passing_yards"],
    min_filter=[2, 2, 20, 20, 40],
    season=season,
    week=week
)

print(comp)