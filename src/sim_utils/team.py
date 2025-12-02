import pandas as pd
from utils.quack import Quack


TEAM_RB_STATS: pd.DataFrame = Quack.fetch_table('team_rushers')
TEAM_QB_STATS: pd.DataFrame = Quack.fetch_table('team_qb_stats')
TEAM_RECEIVER_STATS: pd.DataFrame = Quack.fetch_table('team_receiver_stats')
TEAM_STATS: pd.DataFrame = Quack.fetch_table('team_feats')
OPP_STATS: pd.DataFrame = Quack.fetch_table('opp_feats')
PLAYERS: pd.DataFrame = Quack.query('''select p.*, adv.stuffed,
mediocre, explosive, huge_play, 
first_down, short_first, epa
from player_weekly_agg as p 
left join rusher_adv as adv
on p.gsis_id = adv.gsis_id
and p.game_id = adv.game_id''')

stat_cols: list[str] = [
	"completions",
	"attempts",
	"passing_yards",
	"passing_tds",
	"interceptions",
	"sacks",
	"sack_yards",
	"sack_fumbles",
	"sack_fumbles_lost",
	"passing_air_yards",
	"passing_yards_after_catch",
	"passing_first_downs",
	"passing_epa",
	"passing_2pt_conversions",
	"passing_yac",
	"passing_air_yards",
	"carries",
	"rushing_yards",
	"rushing_tds",
	"rushing_fumbles",
	"rushing_fumbles_lost",
	"rushing_2pt_conversions",
	"receptions",
	"targets",
	"receiving_yards",
	"receiving_tds",
	"receiving_fumbles",
	"receiving_fumbles_lost",
	"receiving_air_yards",
	"receiving_yards_after_catch",
	"receiving_first_downs",
	"receiving_epa",
	"receiving_2pt_conversions",
	"racr",
	"air_yards",
	"yac",
	"target_share",
	"air_yards_share",
	"fantasy_points",
	"fantasy_points_ppr",
]


def fetch_row_or_latest(df: pd.DataFrame, team: str, season: int, week: int, opp=False) -> dict:
	team_field = 'team' if not opp else 'opponent_team'
	try:
		df = df.loc[(df[team_field] == team) & (df.season == season)]
		row = df.loc[(df.week == min(df.week.max(), week))].to_dict(orient="records")[0]
	except Exception as e:
		print(f'No data exists for {team} in {season}')
		raise e
	return row

class Player:
	def __init__(self, player_data: pd.Series):
		self.name: str = player_data["player_display_name"]
		self.id: str = player_data["gsis_id"]
		self.depth_team: int = int(player_data["dense_depth"])
		self.stats: dict = {x: 0 for x in stat_cols}
		self.features: dict = player_data.to_dict()

	def reset_stats(self) -> None:
		self.stats = {stat_name: 0 for stat_name in self.stats}


	def __getattr__(self, name: str):
		# Redirect attribute access to the stats dictionary
		if name in self.stats:
			return self.stats[name]
		elif name in self.features:
			return self.features[name]
		raise AttributeError(
			f"'{self.__class__.__name__}' object has no attribute '{name}'"
		)

	def __setattr__(self, name, value):
		if name in {
			"stats",
			"features",
			"name",
			"id",
			"depth_team",
		}:  # Handle direct attributes
			super().__setattr__(name, value)
		elif name in self.stats:  # Redirect updates to stats dictionary
			self.stats[name] = value
		elif name in self.features:  # Redirect updates to features dictionary
			self.features[name] = value
		else:
			raise AttributeError(f"Cannot set unknown attribute '{name}'")

	def stats_to_dict(self) -> dict:
		out = {}
		out["name"] = self.name
		out["id"] = self.id
		out.update(self.stats)

		return out


class QB(Player):
	def __init__(self, player_data):
		super().__init__(player_data)
		self.features = player_data.to_dict()
		self.position = "QB"

	def __repr__(self) -> str:
		return f"QB:{self.name} has {self.completions} completions for {self.passing_yards} yards"


class RB(Player):
	def __init__(self, player_data):
		super().__init__(player_data)
		self.position = "RB"
		self.features = player_data.to_dict()

	def __repr__(self):
		return (
			f"RB:{self.name} has {self.carries} carries for {self.rushing_yards} yards"
		)


class WR(Player):
	def __init__(self, player_data):
		super().__init__(player_data)
		self.position = "WR"

	def __repr__(self) -> str:
		return f"WR:{self.name} has {self.receptions} receptions for {self.receiving_yards} yards"


class TE(Player):
	def __init__(self, player_data):
		super().__init__(player_data)
		self.position = "TE"

	def __repr__(self) -> str:
		return f"TE:{self.name} has {self.receptions} receptions for {self.receiving_yards} yards"


class K(Player):
	def __init__(self, player_data):
		super().__init__(player_data)
		self.position = "K"

	def __repr__(self) -> str:
		return f"K:{self.name} has placeholder FG and placeholder PAT."


class Team:
	def __init__(self, name: str, season: int, week: int, use_current_injuries=False):
		self.name: str = name
		self.opponent: Team | None = None
		self.score: int = 0
		self.plays: int = 0
		self.features: dict[str, int | str | float] = {"last_rusher_drive": -1, "last_rusher_team": -1}
		self.team_stats = fetch_row_or_latest(TEAM_STATS, self.name, season, week)
		self.opp_stats = fetch_row_or_latest(OPP_STATS, self.name, season, week, opp=True)
		self.roster = PLAYERS.loc[
			(PLAYERS.team == name) & (PLAYERS.season == season)
		]
		self.roster = self.roster.loc[
			(self.roster.week == min(self.roster.week.max(), week))
			#& (self.roster.formation == "Offense")
			#& (self.roster.position.isin(["QB", "WR", "TE", "RB", "K"]))
		].sort_values(by="dense_depth")

		self.QBs: list[QB] = self.build_roster_by_position("QB")
		self.RBs: list[RB] = self.build_roster_by_position("RB")
		self.WRs: list[WR] = self.build_roster_by_position("WR")
		self.TEs: list[TE] = self.build_roster_by_position("TE")
		self.Ks: list[K] = self.build_roster_by_position("K")
		self.players: list[Player] = self.QBs + self.RBs + self.WRs + self.TEs
		self.rb_stats: dict[str, str | int | float] = fetch_row_or_latest(TEAM_RB_STATS, self.name, season, week)

		self.team_receiver_stats: dict[str, str | int | float] = fetch_row_or_latest(
			TEAM_RECEIVER_STATS, self.name, season, week
		)
		self.team_qb_stats: dict[str, str | int | float]  = fetch_row_or_latest(
			TEAM_QB_STATS, self.name, season, week
		)
		self.player_dict: dict[str, Player] = {p.id: p for p in self.players}
	
	def get_player_by_id(self, player_id: str) -> Player:
		return self.player_dict[player_id]

	def build_roster_by_position(self, position: str):
		"""Filter players by position and create player objects."""
		position_data = self.roster[(self.roster["position"] == position)].fillna(0)
		# Create player objects based on position
		players: list[Player] = []
		for _, player_data in position_data.iterrows():
			if position == "WR":
				players.append(WR(player_data))
			elif position == "RB":
				players.append(RB(player_data))
			elif position == "QB":
				players.append(QB(player_data))
			elif position == "TE":
				players.append(TE(player_data))
			elif position == 'K':
				players.append(K(player_data))
		return players

	def get_depth_pos(self, pos: str, depth: int):
		"""input a position and team depth, to get the player
		used to go from ML output -> player object"""
		while depth >= 0:
			if pos == "WR":
				for player in self.WRs:
					if player.depth_team == depth:
						return player
			if pos == "RB":
				for player in self.RBs:
					if player.depth_team == depth:
						return player
			if pos == "TE":
				for player in self.TEs:
					if player.depth_team == depth:
						return player
			if pos == "QB":
				for player in self.QBs:
					if player.depth_team == depth:
						return player
			depth -= 1
		print(pos, depth, self.name, "You want a player that does not exist")
		if pos == "WR":
				return self.WRs[0]
		if pos == "RB":
				return self.RBs[0]
		if pos == "TE":
				return self.TEs[0]
		if pos == "QB":
				return self.QBs[0]
		raise ValueError("You want a player that does not exist")

	def game_results(self, game_id, df=False):
		r = [
			{"team": self.name, "position": x.position, "id": x.id, "sim_id":game_id}
			| x.stats_to_dict()
			for x in self.players
		]
		if df:
			return pd.json_normalize(r)
		return r

	def reset_stats(self):
		[x.reset_stats() for x in self.players]
		self.score = 0
		self.plays = 0

	def __repr__(self) -> str:
		return f"{self.name} has {self.score} points"
	
