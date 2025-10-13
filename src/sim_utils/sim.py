import random
from abc import ABC, abstractmethod
import pandas as pd
import joblib
import numpy as np
import torch
import warnings
from utils.quack import Quack
import yaml
from sim_utils.sim_models import ChooseRusherModel, RushYardsModel, AirYardsModel, YacModel
from time import time

with open("models/feature_config.yaml", "r") as file:
	CONFIG = yaml.safe_load(file)

team_rb_stats = Quack.fetch_table('team_rushers')
team_qb_stats = Quack.fetch_table('team_qb_stats')
team_receiver_stats = Quack.fetch_table('team_receiver_stats')
team_stats = Quack.fetch_table('team_feats')
opp_stats = Quack.fetch_table('opp_feats')
players = Quack.fetch_table('player_weekly_agg')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

RUSH_YARDS_MODEL = RushYardsModel(CONFIG)
AIR_YARDS_MODEL = AirYardsModel(CONFIG)
YAC_MODEL = YacModel(CONFIG)

CLOCK_MODEL = joblib.load("models/clock_model.joblib")

#CHOOSE_RUSHER_MODEL = joblib.load("models/choose_rusher.joblib")
CHOOSE_RUSHER_MODEL = ChooseRusherModel(CONFIG)

CHOOSE_RECEIVER_MODEL = joblib.load("models/choose_receiver.joblib")

COMPLETE_PASS_MODEL = joblib.load("models/complete_pass.joblib")


#QRF_RUN_YARDS = joblib.load("models/rush_yards_qrf.joblib")

stat_cols = [
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
	"pacr",
	"carries",
	"rushing_yards",
	"rushing_tds",
	"rushing_fumbles",
	"rushing_fumbles_lost",
	"rushing_first_downs",
	"rushing_epa",
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
	"target_share",
	"air_yards_share",
	"wopr",
	"special_teams_tds",
	"fantasy_points",
	"fantasy_points_ppr",
]


def fetch_row_or_latest(df, team, season, week):
	try:
		df = df.loc[(df.team == team) & (df.season == season)]
		row = df.loc[(df.week == min(df.week.max(), week))].to_dict(orient="records")[0]
	except Exception as e:
		print(f'No data exists for {team} in {season}')
		raise e
	return row

class Player:
	def __init__(self, d):
		self.name = d["player_display_name"]
		self.id = d["gsis_id"]
		self.depth_team = int(d["dense_depth"])
		self.stats = {x: 0 for x in stat_cols}
		self.stats["air_yards"] = 0
		self.stats["yac"] = 0
		self.features = d.to_dict()

	def reset_stats(self):
		self.stats = {stat_name: 0 for stat_name in self.stats}


	def __getattr__(self, name):
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

	def stats_to_dict(self):
		out = {}
		out["name"] = self.name
		out["id"] = self.id
		out.update(self.stats)

		return out


class QB(Player):
	def __init__(self, d):  # noqa: F811
		super().__init__(d)
		self.features = d.to_dict()
		self.position = "QB"

	def __repr__(self):
		return f"QB:{self.name} has {self.completions} completions for {self.passing_yards} yards"


class RB(Player):
	def __init__(self, d):
		super().__init__(d)
		self.position = "RB"
		self.features = d.to_dict()

	def __repr__(self):
		return (
			f"RB:{self.name} has {self.carries} carries for {self.rushing_yards} yards"
		)


class WR(Player):
	def __init__(self, d):
		super().__init__(d)
		self.position = "WR"

	def __repr__(self):
		return f"WR:{self.name} has {self.receptions} receptions for {self.receiving_yards} yards"


class TE(Player):
	def __init__(self, d):
		super().__init__(d)
		self.position = "TE"

	def __repr__(self):
		return f"TE:{self.name} has {self.receptions} receptions for {self.receiving_yards} yards"


class K(Player):
	def __init__(self, d):
		super().__init__(d)
		self.position = "K"

	def __repr__(self):
		return f"K:{self.name} has placeholder FG and placeholder PAT."


class Team:
	def __init__(self, name: str, season: int, week: int, use_current_injuries=False):
		self.name = name
		self.score = 0
		self.plays = 0
		self.features = {"last_rusher_drive": -1, "last_rusher_team": -1}
		self.team_stats = fetch_row_or_latest(team_stats, self.name, season, week)
		self.opp_stats = fetch_row_or_latest(opp_stats, self.name, season, week)
		self.roster = players.loc[
			(players.team == name) & (players.season == season)
		]
		self.roster = self.roster.loc[
			(self.roster.week == min(self.roster.week.max(), week))
			#& (self.roster.formation == "Offense")
			& (self.roster.position.isin(["QB", "WR", "TE", "RB", "K"]))
		].sort_values(by="dense_depth")

		self.QBs = self.get_players_by_position("QB")
		self.RBs = self.get_players_by_position("RB")
		self.WRs = self.get_players_by_position("WR")
		self.TEs = self.get_players_by_position("TE")
		self.players = self.QBs + self.RBs + self.WRs + self.TEs
		self.rb_stats = fetch_row_or_latest(team_rb_stats, self.name, season, week)

		self.team_receiver_stats = fetch_row_or_latest(
			team_receiver_stats, self.name, season, week
		)
		self.team_qb_stats = fetch_row_or_latest(
			team_qb_stats, self.name, season, week
		)

	def get_players_by_position(self, position: str):
		"""Filter players by position and create player objects."""
		with pd.option_context("future.no_silent_downcasting", True):
			position_data = self.roster[(self.roster["position"] == position)].fillna(0)
		# Create player objects based on position
		players = []
		for _, player_data in position_data.iterrows():
			if position == "WR":
				players.append(WR(player_data))
			elif position == "RB":
				players.append(RB(player_data))
			elif position == "QB":
				players.append(QB(player_data))
			elif position == "TE":
				players.append(TE(player_data))
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

	def __repr__(self):
		return f"{self.name} has {self.score} points"
	

class Play(ABC):
	def __init__(self, game):
		self.game = game
		self.clock_cols = CONFIG["clock_cols"]
		self.play_context = {}
		self.play_data = {'incomplete_pass':0,
						  'out_of_bounds':0,
					  'player':None,
					  'timeout': 0,
					  'sp': 0}

	@abstractmethod
	def execute_play(self, team):
		return
	
	def log_play(self, play_type, yds, play_time_elapsed=0, verbose=False):
		"""Logs the context of the game state at each play."""
		play_data = {
			"play_type": play_type,
			"yards_gained": yds,
			"player": self.play_data['player'],
			"play_time_elapsed": play_time_elapsed,
		}
		play_data.update(self.game.game_context)
		if verbose:
			print(
				f'{self.game.possession.name} {play_type} for {yds} yards, {self.game.pbp[-1]['yardline_100']} yd line,'
				+ f' {self.game.pbp[-1]['ydstogo']} yds to go on {self.game.pbp[-1]['down']} down.'
				+ f' {self.game.pbp[-1]['quarter_seconds_remaining'] // 60}:{self.game.pbp[-1]['quarter_seconds_remaining']  % 60} left'
			)
		self.game.pbp.append(play_data)
		return play_data

	
	def sample_clock(self, play_type):
		"""this is the time from the previous play, until the next play starts."""
		raw_features = self.collect_features(
			{"next_play": self.game.int_from_play[play_type]},
			 self.game.pbp[-1],
			self.play_data
		)
		raw_features['play_type_enc'] = self.game.int_from_play[self.game.pbp[-1]['play_type']]
		features = np.array([[raw_features[key] for key in self.clock_cols]])
		t = CLOCK_MODEL.predict(features).item()+5
		return t


	def update_game_state(self, team, yards):
		if self.play_type not in ["field_goal", "punt", "pos_timeout", "def_timeout"]:
			self.game.ydstogo -= yards
			self.game.ball_position -= yards
			self.game.td_check(team)
			self.game.check_downs(team)
	
	def orchestrate(self, team):
		yards = self.execute_play(team)
		self.update_game_state(team, yards)

	def collect_features(self, *argv):
		features = {}
		features.update(self.game.game_context)
		for arg in argv:
			features.update(arg)
		return features


class RunPlay(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='run'
		self.rusher_idx_to_pos = CONFIG["rusher_idx_to_pos"]
		self.rush_yard_cols = CONFIG["rush_yard_cols"]

	def choose_rusher(self, team):
		raw_features = self.collect_features(
			team.rb_stats,
			team.features,
			self.game.defending.opp_stats,
		)
		rusher_idx = CHOOSE_RUSHER_MODEL.predict(raw_features)
		pos, depth = self.rusher_idx_to_pos[rusher_idx].split("_")
		player = team.get_depth_pos(pos, int(depth))
		team.features["last_rusher_team"] = rusher_idx
		team.features["last_rusher_drive"] = rusher_idx
		return player

	def sample_run_yards(self, team, player):
		raw_features = self.collect_features(
			player.features,
			team.team_stats,
			team.opponent.opp_stats,
		)
		sample = RUSH_YARDS_MODEL.predict(raw_features)
		return min((sample, raw_features["yardline_100"]))

	def execute_play(self, team):
		self.player = self.choose_rusher(team)
		yds = self.sample_run_yards(team, self.player)
		yds = min(yds, self.game.ball_position)
		return yds

	def update_game_state(self, team, yds):
			#super().update_game_state()
			self.player.carries += 1
			self.player.rushing_yards += yds
			self.game.player = self.player.name
			return

	def sample_run_yards_quant(self, model, team, player):
		raw_features = self.collect_features(
			player.features,
			team.team_stats,
			team.opponent.opp_stats,
		)
		x = [raw_features[key] for key in self.rush_yard_cols]
		x = np.array([x])
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=UserWarning)
			quantile = np.random.randint(0,100)/100
			sample = model.predict(x, quantiles=quantile)
		return round(sample[0])
	

class PassPlay(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='pass'

		self.choose_receiver_cols = CONFIG["choose_receiver_cols"]
		self.air_yards_cols = CONFIG["air_yards_cols"]
		self.receiver_idx_to_pos = CONFIG["receiver_idx_to_pos"]
		self.complete_pass_cols = CONFIG["complete_pass_cols"]
		


	def sample_air_and_yac(self, team, player):
		raw_features = self.collect_features(
			player.features,
			team.team_stats,
			team.opponent.opp_stats,
		)
		air_yards = AIR_YARDS_MODEL.predict(raw_features)
		if air_yards >= self.game.ball_position:  # touchdown at catch
			return self.game.ball_position, 0
		
		raw_features['air_yards'] = air_yards
		raw_features['catch_yardline'] = self.game.ball_position - air_yards
		raw_features['air_td'] = air_yards >= self.game.ball_position
		raw_features['air_fd'] = air_yards >= self.game.ydstogo
		yac = YAC_MODEL.predict(raw_features)
		yac = min(yac, (self.game.ball_position - air_yards))
		self.play_context.update({'air_yards':air_yards,
						  'yac': yac})
		return air_yards, yac

	def sample_completion(self, qb, receiver, team, air_yards):
		raw_features = self.collect_features(
			receiver.features,
			team.team_stats,
			team.opponent.opp_stats,
		)
		raw_features["air_yards"] = air_yards
		qb_features = {(key + "_qb"): value for key, value in qb.features.items()}
		raw_features.update(qb_features)
		features = [raw_features[key] for key in self.complete_pass_cols]
		preds = COMPLETE_PASS_MODEL.predict_proba([features])
		receiver = np.random.choice(len(preds[0]), p=preds[0])
		return np.random.choice(len(preds[0]), p=preds[0])
	
	def get_receiver(self, team):
		raw_features = self.collect_features(
			team.team_receiver_stats,
			team.features,
		)
		features = [raw_features[key] for key in self.choose_receiver_cols]
		preds = CHOOSE_RECEIVER_MODEL.predict_proba([features])
		receiver = np.random.choice(len(preds[0]), p=preds[0])
		pos, depth = self.receiver_idx_to_pos[receiver].split("_")
		receiver = team.get_depth_pos(pos, int(depth))
		return receiver

	def execute_play(self, team):
		passer = team.QBs[0]
		receiver = self.get_receiver(team)
		passer.attempts += 1
		receiver.targets += 1
		air_yards, yac = self.sample_air_and_yac(
			team, receiver
		)

		if self.sample_completion(passer, receiver, team, air_yards):
			passer.completions += 1
			receiver.receptions += 1
			yds = air_yards + yac
			receiver.air_yards += air_yards
			receiver.yac += yac
			receiver.receiving_yards += yds
			passer.passing_yards += yds
		else:
			yds = 0
		self.play_data['player'] = receiver.name
		return yds


class FieldGoal(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='field_goal'

	def execute_play(self, team):
		result = random.randint(0, 100)
		if result > (10 + (1.7 * self.game.ball_position)):
			team.score += 3
			self.game.switch_poss()
			self.game.ball_position = 65
			# print(f'{team.name} scored a FG')
		else:
			# print(f'{team.name} missed FG')
			self.game.switch_poss()
		self.game.player = None
		return 0


class Punt(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='punt'

	def execute_play(self, team):
		return 0
	
	def update_game_state(self, team, yards):
		self.game.switch_poss()
		self.game.ball_position += random.randint(45, 60)
		if self.game.ball_position >= 100:
			self.game.ball_position = 20
		self.game.player = None
		return 0

class Kneel(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='qb_kneel'
	def execute_play(self, team):
		# Implementation of qb kneel play
		# print("QB kneel executed.")
		return -1


class Spike(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='qb_spike'

	def execute_play(self, team):
		# Implementation of qb spike play
		# print("QB spike executed.")
		return -1


class PosTimeout(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='pos_timeout'
	def execute_play(self, team):
		self.game.possession.timeouts -= 1
		#print(f"TIMEOUT! {self.game.possession.timeouts} remaining")
		return 0


class DefTimeout(Play):
	def __init__(self, game):
		super().__init__(game)
		self.play_type='def_timeout'
	def execute_play(self, team):
		self.game.defending.timeouts -= 1
		#print(f"def TIMEOUT!, {self.game.defending.timeouts} remaining")
		return 0

class GameState:
	def __init__(self, away, home, **kwargs):
		"""Set initial values when instantiating a GameState object"""
		self.home = home
		self.away = away
		home.opponent = self.away
		away.opponent = self.home
		home.spread_line = kwargs.get("spread_line", -3)
		away.spread_line = -1 * self.home.spread_line
		self.total_line = kwargs.get("total_line", 42)
		self.run_or_pass = joblib.load("models/run_or_pass.joblib")
		self.run_or_pass_cols = CONFIG["run_or_pass_cols"]
		self.play_encoding = CONFIG["play_encoding"]
		self.int_from_play = {v: k for k, v in self.play_encoding.items()}
		self.wind = kwargs.get("wind", random.randint(0, 10))
		self.temp = kwargs.get("temp", random.randint(40, 90))
		self.play_functions = {
			"field_goal": FieldGoal(self),
			"no_play": RunPlay(self),
			"pass": PassPlay(self),
			"punt": Punt(self),
			"qb_kneel": Kneel(self),
			"qb_spike": Spike(self),
			"run": RunPlay(self),
			"pos_timeout": PosTimeout(self),
			"def_timeout": DefTimeout(self),
		}
		self.reset_game()
		self.game_context = self.get_game_state()

	def reset_game(self)-> None: 
		"""Reset values to starting. Used when initializing GameState, and between consecutive sims"""
		self.quarter = 1
		self.possession = self.home
		self.defending = self.away
		self.down = 1
		self.ydstogo = 10
		self.ball_position = 65  # Yardline (0-100), 0 is score, 100 is safety
		self.clock = 900  # Seconds in the current quarter (15 mins = 900 seconds)
		self.drive = 0
		self.pbp = []
		self.player = None
		self.home.timeouts = 3
		self.away.timeouts = 3
		self.game_id = int(time() * 1000) 
		self.home.reset_stats()
		self.away.reset_stats()

	def switch_poss(self):
		self.possession.features["last_rusher_drive"] = -1
		self.possession = self.away if self.possession == self.home else self.home
		self.defending = self.possession.opponent
		self.ball_position = 100 - min(self.ball_position, 99)
		self.down = 1
		self.ydstogo = min(10, self.ball_position)
		self.drive += 1
		return

	def kickoff(self):
		self.switch_poss()
		self.ball_position = 65
		self.log_play("kickoff", 0)
		pass

	def start_game(self):
		self.lost_kickoff = random.choice((self.home, self.away))
		self.possession = self.lost_kickoff
		self.kickoff()
		self.game_context = self.get_game_state()
		# print(f"{self.possession.name} has won the kickoff")
		self.log_play("kickoff", 0)

	def collect_features(self, *argv):
		features = {}
		features.update(self.game_context)
		for arg in argv:
			features.update(arg)
		return features

	def get_game_state(self):
		"""Logs the context of the game state at each play."""
		play_data = {
			"possession": self.possession.name,
			"quarter": self.quarter,
			"down": self.down,
			"is_first_down": self.down == 1,
			"ydstogo": self.ydstogo,
			"goal_to_go": int(self.ball_position < 10),
			"yardline_100": self.ball_position,
			"total_home_score": self.home.score,
			"total_away_score": self.away.score,
			"posteam_score": self.possession.score,
			"defteam_score": self.defending.score,
			"score_differential": (self.possession.score - self.defending.score),
			"wind": self.wind,
			"temp": self.temp,
			"quarter_seconds_remaining": self.clock,
			"half_seconds_remaining": self.clock + (900 * (self.quarter % 2)),
			"game_seconds_remaining": self.clock + (900 * (4 - self.quarter)),
			"drive": self.drive,
			"spread_line": self.possession.spread_line,
			"total_line": self.total_line,
			"posteam_timeouts_remaining": self.possession.timeouts,
			"defteam_timeouts_remaining": self.defending.timeouts,
		}
		return play_data

	def play(self, team):
		self.game_context = self.get_game_state()
		team.plays += 1
		raw_features = self.collect_features(
			team.team_stats,
			self.defending.opp_stats,
		)
		features = [raw_features[key] for key in self.run_or_pass_cols]
		preds = self.run_or_pass.predict_proba([features])
		if self.possession.timeouts == 0:
			preds[:, 7] = 0
			preds /= preds.sum()
		if self.defending.timeouts == 0:
			preds[:, 8] = 0
			preds /= preds.sum()
		play_type_int = np.random.choice(len(preds[0]), p=preds[0])
		play_type = self.play_encoding.get(play_type_int, 1)
		#play_time_elapsed = self.play_functions[play_type].sample_clock(play_type)
		#self.clock -= play_time_elapsed
		yds = self.play_functions[play_type].orchestrate(team)
		#self.log_play(play_type, yds, play_time_elapsed) ########TODO

	def log_play(self, play_type, yds,play_time_elapsed, verbose=False):
		"""Logs the context of the game state at each play."""
		play_data = {
			"play_type": play_type,
			"yards_gained": yds,
			"player": self.player,
			"play_time_elapsed": play_time_elapsed
		}
		play_data.update(self.game_context)
		if verbose:
			print(
				f"{self.possession.name} {play_type} for {yds} yards, {self.pbp[-1]['yardline_100']} yd line,"
				+ f" {self.pbp[-1]['ydstogo']} yds to go on {self.pbp[-1]['down']} down."
				+ f" {self.pbp[-1]['quarter_seconds_remaining'] // 60}:{self.pbp[-1]['quarter_seconds_remaining'] % 60} left"
			)
		self.pbp.append(play_data)
		return play_data

	def td_check(self, team):
		if self.ball_position <= 0:
			team.score += 7
			self.pbp[-1]
			self.kickoff()
			# print(f'{team.name} scored a TD')
		return

	def check_downs(self, team):
		if self.ydstogo <= 0:
			self.ydstogo = 10
			self.down = 1
		elif self.down == 4:
			self.switch_poss()
		else:
			self.down += 1

	def play_quarter(self):
		self.clock = 900
		if self.quarter == 3:
			self.kickoff()
			self.possession = self.lost_kickoff
		if self.quarter == 3:
			self.home.timeouts = 3
			self.away.timeouts = 3
		while self.clock > 0:
			self.play(self.possession)
		self.quarter += 1
		# print(f"{self.home.name}:{self.home.score}")
		# print(f"{self.away.name}:{self.away.score}")

	def play_game(self):
		self.reset_game()
		self.start_game()
		while self.quarter <= 4:
			self.play_quarter()
			# print(self.quarter)
		if self.home.score > self.away.score:
			print(f"{self.home.name} has won {self.home.score} - {self.away.score}")
		else:
			print(f"{self.away.name} has won {self.away.score} - {self.home.score}")

	def game_results(self, df=False):
		res1 = self.home.game_results(self.game_id,df=False)
		res2 = self.away.game_results(self.game_id,df=False)
		res = res1 + res2
		if df:
			return pd.DataFrame(res)
		return res