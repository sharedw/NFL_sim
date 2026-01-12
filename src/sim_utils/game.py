from __future__ import annotations
from sim_utils.GameState import GameStateDict
from time import time 
import pandas as pd
import random
import numpy as np
from sim_utils.config import CONFIG
from utils.quack import Quack
from sim_utils.plays import play_registry
from sim_utils.team import Team
from sim_utils.play_result import PlayResult
from sim_utils.GameModels import ClockModel
import joblib

CLOCK_MODEL = ClockModel(CONFIG)

class GameState:
	def __init__(self, away: Team, home: Team, **kwargs):
		#self.event_log = EventLog()
		self.home: Team = home
		self.away: Team = away
		home.opponent = self.away
		away.opponent = self.home
		home.spread_line = kwargs.get("spread_line", -3)
		away.spread_line = -1 * self.home.spread_line
		self.total_line = kwargs.get("total_line", 42)
		self.run_or_pass = joblib.load("models/run_or_pass.joblib")
		self.run_or_pass_cols = CONFIG["run_or_pass_cols"]
		self.play_encoding = CONFIG["play_encoding"]
		self.play_decoding = CONFIG["play_decoding"]
		self.wind = kwargs.get("wind", random.randint(0, 10))
		self.temp = kwargs.get("temp", random.randint(40, 90))
		self.play_functions = {name: cls() for name, cls in play_registry.items()}
		self.last_play = PlayResult()
		self.reset_game()
		self.game_context = self.get_game_state()
		assert home.opponent is not None
		assert away.opponent is not None

	def reset_game(self):
		self.quarter = 1
		self.possession: Team = self.home
		self.defending: Team = self.away
		self.down = 1
		self.ydstogo = 10
		self.ball_position: int = 65  # Yardline (0-100), 0 is score, 100 is safety
		self.clock = 900  # Seconds in the current quarter (15 mins = 900 seconds)
		self.drive = 0
		self.pbp = []
		self.home.timeouts = 3
		self.away.timeouts = 3
		self.game_id = int(time() * 1000) 
		self.home.reset_stats()
		self.away.reset_stats()

	def switch_poss(self) -> None:
		self.possession.features["last_rusher_drive"] = -1
		self.possession = self.away if self.possession == self.home else self.home
		self.defending: Team = self.possession.opponent
		self.ball_position = 100 - min(self.ball_position, 99)
		self.down = 1
		self.ydstogo = min(10, self.ball_position)
		self.drive += 1
		return

	def start_game(self):
		self.lost_kickoff: Team = random.choice((self.home, self.away))
		self.possession: Team = self.lost_kickoff
		self.sim_one_play(self.possession, kickoff=True)


	def collect_features(self, *argv) -> dict:
		features = {}
		for arg in argv:
			if isinstance(arg, PlayResult):
				features.update(arg.to_dict())
			elif arg:
				features.update(arg)
		return features
	
	def sample_clock(self, game_context, next_play_type) -> int:
		
		raw_features = self.collect_features(
			game_context, 
			self.last_play, 
			{'next_play_type': next_play_type,
			'play_type_enc': CONFIG['play_decoding'].get(next_play_type, 0),
			    'is_field_goal': next_play_type == 'field_goal',
				'is_no_play': next_play_type == 'no_play',
				'is_punt': next_play_type == 'punt',
				'is_qb_kneel': next_play_type =='qb_kneel',
				'is_qb_spike': next_play_type == 'qb_spike',
				'is_run': next_play_type == 'run',
				'is_pass': next_play_type == 'pass',
				'is_timeout': next_play_type == 'def_timeout' or next_play_type == 'pos_timeout'
			}
		)
		play_duration = CLOCK_MODEL.predict(raw_features)
		return min(max(0, int(play_duration)), 60)

	def get_game_state(self) -> GameStateDict:
		"""Fetches the current context of the game state. Can be used as model input or for logging"""
		return {
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

	def call_play(self, team: Team, game_context: GameStateDict) -> str:
		"""This uses an XGBoost model to predict what play type will be ran next."""
		assert team.opponent is not None
		raw_features = self.collect_features(
			team.team_stats,
			team.opponent.opp_stats,
			game_context
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
		play_type = self.play_decoding.get(play_type_int, 1)
		if play_type == 'no_play':
			play_type = 'run'
		return play_type
	
	def update_game_state(self, team: Team,  play_result: PlayResult):
		#self.clock -= 25
		assert play_result.play_type is not None

		if play_result.play_type in ['run','pass']:
			yards = play_result.yards
			self.ydstogo -= yards
			self.ball_position -= yards
			self.td_check(team)
			self.check_downs(team)
			return

		elif play_result.play_type == 'qb_kneel':
			yards = random.randint(1,2)
			self.ydstogo -= yards
			self.ball_position -= yards
			self.check_downs(team)
			return

		elif play_result.play_type == 'qb_spike':
			self.check_downs(team)
			return
		
		elif play_result.play_type == 'field_goal':
			result = np.random.randint(0, 100)
			if result > (10 + (1.7 * self.ball_position)):
				team.score += 3
				self.switch_poss()
				self.ball_position = 65
			else:
				self.switch_poss()
			return
		
		elif play_result.play_type == 'punt':
			self.ball_position -= np.random.randint(35, 60)
			self.switch_poss()
			if self.ball_position >= 100:
				self.ball_position = 20
			return 

		elif play_result.play_type == 'pos_timeout':
			self.possession.timeouts -= 1
			return
		
		elif play_result.play_type == 'def_timeout':
			self.defending.timeouts -= 1
			return
		
		elif play_result.play_type == 'kickoff':
			self.switch_poss()
			self.ball_position = 65 #TODO add kick model
			return
		
		return

	def sim_one_play(self, team: Team, kickoff=False) -> None:
		"""This is the core functionality of the sim. This simulates one play of a given type.
		Game context data is passed in to feed the models their features for prediction."""
		team.plays += 1
		self.game_context = self.get_game_state()
		if not kickoff:
			play_type = self.call_play(team, self.game_context)
		else:
			play_type='kickoff'
		if self.pbp:
			play_duration = self.sample_clock(self.game_context, play_type )
			self.clock -= play_duration
		self.game_context = self.get_game_state()
		play_result = self.play_functions[play_type].orchestrate(team, self.game_context)
		self.update_game_state(team, play_result)
		self.last_play = play_result
		self.log_play(play_result)
		return 


	def log_play(self, play_result: PlayResult) -> PlayResult:
		"""Logs the context of the game state at each play."""
		"""play_data = {
			"play_type": play_result.play_type,
			"yards_gained": play_result.yards,
			"player": play_result.get('rusher', play_result.get('receiver',0)),
			"play_time_elapsed": 'play_time_elapsed'
		}"""
		play_result.to_dict().update(self.game_context)
		play_log = play_result
		self.pbp.append(play_log)
		return play_log

	def td_check(self, team):
		if self.ball_position <= 0:
			team.score += 7
			self.sim_one_play(team, kickoff=True)
			# print(f'{team.name} scored a TD')
		if self.ball_position >= 100:
			team.opponent.score += 2
			self.sim_one_play(team.opponent, kickoff=True)
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
			self.possession = self.lost_kickoff
			self.sim_one_play(self.possession, kickoff=True)
		if self.quarter == 3:
			self.home.timeouts = 3
			self.away.timeouts = 3
		while self.clock > 0:
			self.sim_one_play(self.possession)
		self.quarter += 1
		# print(f"{self.home.name}:{self.home.score}")
		# print(f"{self.away.name}:{self.away.score}")

	def play_game(self, save_results: bool=False):
		self.reset_game()
		self.start_game()
		while self.quarter <= 4:
			self.play_quarter()
			# print(self.quarter)
		if self.home.score > self.away.score:
			print(f"{self.home.name} has won {self.home.score} - {self.away.score}")
		else:
			print(f"{self.away.name} has won {self.away.score} - {self.home.score}")
		if save_results:
			df = self.game_results(df=True) #noqa
			Quack.query("""
				CREATE TABLE IF NOT EXISTS sim_results AS
				SELECT * FROM df WHERE 1=0
				""")
			Quack.query("""
				INSERT INTO results
				SELECT * FROM df
				""")


	def game_results(self, df=False) -> pd.DataFrame | dict:
		res1 = self.home.game_results(self.game_id,df=False)
		res2 = self.away.game_results(self.game_id,df=False)
		res = res1 + res2
		if df:
			return pd.DataFrame(res)
		return res