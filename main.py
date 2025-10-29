from sim_utils.game import GameState
from sim_utils.team import Team


t1 = Team('TB', 2025, 7)
t2 = Team('KC', 2025, 7)

game = GameState(t1, t2)

game.reset_game()

game.play_game()

df =game.game_results(df=True)[['name','team','carries', 'receptions','targets','rushing_yards','receiving_yards']]
df = df[df.ne(0).sum(axis=1) > 2] #print only players with stats
print(df)


