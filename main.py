from sim_utils.game import GameState
from sim_utils.team import Team


t1 = Team('DET', 2025, 9)
t2 = Team('MIN', 2025, 9)

game = GameState(t1, t2)

game.reset_game()

game.play_game()

df =game.game_results(df=True)[['name','team','carries', 'receptions','targets','rushing_yards','receiving_yards']]
df = df[df.ne(0).sum(axis=1) > 2] #print only players with stats
print(df)


print(t2.Ks)