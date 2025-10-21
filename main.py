from sim_utils.game import GameState
from sim_utils.team import Team


t1 = Team('DET', 2025, 6)
t2 = Team('MIN', 2025, 6)

game = GameState(t1, t2)

game.reset_game()

game.play_game()

df =game.game_results(df=True)[['name','team','carries', 'receptions','targets','rushing_yards','receiving_yards']]

print(df)