from sim_utils.game import GameState
from sim_utils.team import Team
from utils.quack import Quack

def sim_games(team1_name: str, team2_name: str, season:int, week:int, n_games:int=100) -> list:
	result = []
	
	t1 = Team(team1_name, season, week)
	t2 = Team(team2_name, season, week)

	game = GameState(t1, t2)

	for _ in range(n_games):
		game.start_game()
		game.play_game()
		result += game.game_results()
		game.reset_game()


	df =game.game_results(df=True)[['name','team','carries', 'receptions','targets','rushing_yards','receiving_yards']]
	df = df[df.ne(0).sum(axis=1) > 2] #print only players with stats
	print(df)
	return result

def get_actual_results(team1_name: str, team2_name: str, season:int, week:int) -> pd.DataFrame:
	Quack.query("""
	select * from weekly where season = {season}
			 and week = {week}
			 and ((home = {team1_name} and away = {team2_name})
			 or 
			 (home = {team2_name} and away = {team1_name}))""")