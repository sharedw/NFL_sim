from abc import ABC, abstractmethod
import numpy as np
import joblib
from sim_utils.config import CONFIG
from sim_utils.sim_models import (
    ChooseRusherModel,
    RushYardsModel,
    AirYardsModel,
    YacModel,
    OOBModel,
    ClockModel,
)
from sim_utils.team import Team, Player
from sim_utils.play_result import PlayResult

RUSH_YARDS_MODEL = RushYardsModel(CONFIG)
AIR_YARDS_MODEL = AirYardsModel(CONFIG)
YAC_MODEL = YacModel(CONFIG)
CLOCK_MODEL = ClockModel(CONFIG)
OOB_MODEL = OOBModel(CONFIG)
CHOOSE_RUSHER_MODEL = ChooseRusherModel(CONFIG)

CHOOSE_RECEIVER_MODEL = joblib.load("models/choose_receiver.joblib")

COMPLETE_PASS_MODEL = joblib.load("models/complete_pass.joblib")


class Play(ABC):
    def __init__(self):
        self.play_type = None

    @abstractmethod
    def execute_play(self, team: Team, game_context: dict) -> PlayResult:
        raise NotImplementedError

    def sample_oob(self, game_context, play_result) -> int:
        raw_features = self.collect_features(game_context, play_result)
        oob = OOB_MODEL.predict(raw_features)
        return oob

    def orchestrate(self, team: Team, game_context: dict) -> PlayResult:
        play_result = self.execute_play(team, game_context)
        play_result.play_type = self.play_type
        oob = self.sample_oob(game_context, play_result)
        play_result.out_of_bounds = oob
        return play_result

    def collect_features(self, *argv) -> dict:
        features = {}
        for arg in argv:
            if isinstance(arg, PlayResult):
                features.update(arg.to_dict())
            else:
                features.update(arg)
        return features


class RunPlay(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "run"
        self.rusher_idx_to_pos = CONFIG["rusher_idx_to_pos"]
        self.rush_yard_cols = CONFIG["rush_yard_cols"]

    def choose_rusher(self, team, game_context) -> Player:
        raw_features = self.collect_features(
            team.rb_stats, team.features, team.opponent.opp_stats, game_context
        )
        rusher_idx = CHOOSE_RUSHER_MODEL.predict(raw_features)
        pos, depth = self.rusher_idx_to_pos[rusher_idx].split("_")
        player = team.get_depth_pos(pos, int(depth))
        team.features["last_rusher_team"] = rusher_idx
        team.features["last_rusher_drive"] = rusher_idx
        return player

    def sample_run_yards(self, team, game_context, player) -> int:
        raw_features = self.collect_features(
            player.features, team.team_stats, team.opponent.opp_stats, game_context
        )
        sample = RUSH_YARDS_MODEL.predict(raw_features)
        return min((sample, raw_features["yardline_100"]))

    def update_player_stats(self, team: Team, play_result: PlayResult):
        assert play_result.rusher_id is not None
        player = team.get_player_by_id(play_result.rusher_id)
        player.carries += 1
        player.rushing_yards += play_result.yards
        return

    def execute_play(self, team: Team, game_context: dict):
        rusher = self.choose_rusher(team, game_context)
        yds = self.sample_run_yards(team, game_context, rusher)
        yds = min(yds, game_context["yardline_100"])
        play_result = PlayResult(
            yards=int(yds), rusher_id=rusher.id, rusher=rusher.name
        )
        self.update_player_stats(team, play_result)
        return play_result


class PassPlay(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "pass"
        self.choose_receiver_cols: list[str] = CONFIG["choose_receiver_cols"]
        self.air_yards_cols: list[str] = CONFIG["air_yard_cols"]
        self.receiver_idx_to_pos: list[str] = CONFIG["receiver_idx_to_pos"]
        self.complete_pass_cols: list[str] = CONFIG["complete_pass_cols"]

    def sample_air_and_yac(self, team, player, game_context):
        raw_features = self.collect_features(
            player.features, team.team_stats, team.opponent.opp_stats, game_context
        )
        air_yards: int = AIR_YARDS_MODEL.predict(raw_features)
        if air_yards >= game_context["yardline_100"]:  # touchdown at catch
            return game_context["yardline_100"], 0

        raw_features["air_yards"] = air_yards
        raw_features["catch_yardline"] = game_context["yardline_100"] - air_yards
        is_td = air_yards >= game_context["yardline_100"]
        raw_features["air_fd"] = air_yards >= game_context["ydstogo"]
        raw_features["air_td"] = is_td
        if is_td:
            yac = 0
        else:
            yac = YAC_MODEL.predict(raw_features)
            yac = min(yac, (game_context["yardline_100"] - air_yards))
        return air_yards, yac

    def sample_completion(self, qb, receiver, team, air_yards, game_context):
        raw_features = self.collect_features(
            receiver.features, team.team_stats, team.opponent.opp_stats, game_context
        )
        raw_features["air_yards"] = air_yards
        qb_features = {(key + "_qb"): value for key, value in qb.features.items()}
        raw_features.update(qb_features)
        features = [raw_features[key] for key in self.complete_pass_cols]
        preds = COMPLETE_PASS_MODEL.predict_proba([features])
        receiver = np.random.choice(len(preds[0]), p=preds[0])
        return np.random.choice(len(preds[0]), p=preds[0])

    def get_receiver(self, team: Team, game_context: dict):
        raw_features = self.collect_features(
            team.team_receiver_stats, team.features, game_context
        )
        features = [raw_features[key] for key in self.choose_receiver_cols]
        preds = CHOOSE_RECEIVER_MODEL.predict_proba([features])
        receiver = np.random.choice(len(preds[0]), p=preds[0])
        pos, depth = self.receiver_idx_to_pos[receiver].split("_")
        receiver = team.get_depth_pos(pos, int(depth))
        return receiver

    def update_player_stats(self, team: Team, play_result: PlayResult):
        assert play_result.passer_id is not None
        assert play_result.receiver_id is not None
        passer = team.get_player_by_id(play_result.passer_id)
        receiver = team.get_player_by_id(play_result.receiver_id)
        passer.attempts += 1
        receiver.targets += 1
        if play_result.complete_pass:
            passer.completions += 1
            receiver.receptions += 1
            receiver.air_yards += play_result.air_yards
            receiver.yac += play_result.yac
            receiver.receiving_yards += play_result.yards
            passer.passing_yards += play_result.yards
            passer.passing_air_yards += play_result.air_yards
            passer.passing_yac += play_result.yac
        return

    def execute_play(self, team: Team, game_context: dict) -> PlayResult:
        passer = team.QBs[0]
        receiver = self.get_receiver(team, game_context)

        air_yards, yac = self.sample_air_and_yac(team, receiver, game_context)
        completion = self.sample_completion(
            passer, receiver, team, air_yards, game_context
        )
        if completion:
            yds = air_yards + yac
        else:
            yds = 0
            yac = 0

        play_result = PlayResult(
                complete_pass=completion,
                incomplete_pass= not completion,
                yards=yds,
                air_yards=air_yards,
                yac=yac,
                passer_id= passer.id,
                receiver_id= receiver.id if receiver else None,
                passer= passer.name,
                receiver= receiver.name if receiver else None,
            
        )
        self.update_player_stats(team, play_result)
        return play_result


class FieldGoal(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "field_goal"

    def update_player_stats(
        self, team: Team, play_result: dict
    ):  # TODO: add play_result stuff
        kicker = "dork"  # team.get_player_by_id(play_result['passer_id']) #noqa
        return

    def execute_play(self, team: Team, game_context: dict):
        play_result = PlayResult(
            yards= 0, kicker= "dork")
        self.update_player_stats(team, play_result)
        return play_result


class Punt(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "punt"

    def execute_play(self, team: Team, game_context: dict):
        play_result = PlayResult(yards=0)
        self.update_game_state(team, play_result)
        return play_result

    def update_game_state(self, team, yards):
        return


class Kickoff(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "kickoff"

    def execute_play(self, team: Team, game_context: dict) -> PlayResult:
        play_result = PlayResult(yards=0)
        self.update_game_state(team, play_result)
        return play_result

    def update_game_state(self, team, yards):
        return


class Kneel(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "qb_kneel"

    def update_player_stats(self, team, game_context):
        pass

    def execute_play(self, team: Team, game_context: dict) -> PlayResult:
        # Implementation of qb kneel play
        # print("QB kneel executed.")
        return PlayResult(yards=0)


class Spike(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "qb_spike"

    def update_player_stats(self, team, game_context):
        pass

    def execute_play(self, team: Team, game_context: dict) -> PlayResult:
        # Implementation of qb spike play
        # print("QB spike executed.")
        return  PlayResult(yards=0)


class PosTimeout(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "pos_timeout"

    def execute_play(self, team: Team, game_context: dict):
        # print(f"TIMEOUT! {self.game.possession.timeouts} remaining")
        return {"yards": 0}


class DefTimeout(Play):
    def __init__(self):
        super().__init__()
        self.play_type = "def_timeout"

    def execute_play(self, team: Team, game_context: dict):
        # print(f"def TIMEOUT!, {self.game.defending.timeouts} remaining")
        return {"yards": 0}


play_registry = {
    "run": RunPlay,
    "pass": PassPlay,
    "field_goal": FieldGoal,
    "punt": Punt,
    "qb_kneel": Kneel,
    "qb_spike": Spike,
    "pos_timeout": PosTimeout,
    "def_timeout": DefTimeout,
    "kickoff": Kickoff,
}
