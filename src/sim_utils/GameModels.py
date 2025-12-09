import joblib
import torch
from sim_utils.modeling import maskedModel, maskedModelYac
from abc import ABC, abstractmethod
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class GameModel(ABC):
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = self._load_model()  # Force subclasses to implement
        self.model_name = None
    
    @abstractmethod
    def _load_model(self):
        """Load the specific model - must be implemented by subclass"""
        pass
    
    @abstractmethod
    def predict(self, input):
        """Make prediction - must be implemented by subclass"""
        pass
    
    
    def _fetch_model_input(self, *feature_sources, ignore_missing=False) -> list:
        combined = {}
        for source in feature_sources:
            if source is not None:
                combined.update(source)
        if not ignore_missing:
            return [combined[col] for col in self.feature_cols]
        else:
            input = [combined.get(col, 0) for col in self.feature_cols]
            missing_features = [x for x in self.feature_cols if x not in combined.keys()]
            if missing_features:
                print(self.model_name, 'missing input cols:', missing_features)
            return input

class ChooseRusherModel(GameModel):
    """returns position and depth of rusher"""
    def __init__(self, config):
        super().__init__(config["choose_rusher_cols"])
        self._load_model()
        self.model_name = 'ChooseRusherModel'

    def _load_model(self) -> None:
        self.model = joblib.load("models/choose_rusher.joblib")

    def predict(self, input: dict) -> int:
        features = self._fetch_model_input(input)
        preds = self.model.predict_proba([features])
        rusher_idx: int = np.random.choice(len(preds[0]), p=preds[0])
        return rusher_idx
    

class RushYardsModel(GameModel):
    """returns yards gained"""
    def __init__(self, config):
        super().__init__(config["rush_yard_cols"])
        self._load_model()
        self.model_name: str = 'RushYardsModel'

    def _load_model(self) -> None:
        model_path: str = "models/run_yards_gained.pt"
        self.model = maskedModel(n_in=13, n_hidden=128, n_out=140).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))


    def predict(self, input: dict) -> int:
        
        features = self._fetch_model_input(input)
        x = torch.tensor(features).float().to(device)
        with torch.no_grad():
            preds = self.model(x.reshape(1, -1))[0]
            preds = torch.softmax(preds, 0)
        sample = int(torch.multinomial(preds, 1).item()) - 40
        return sample

class AirYardsModel(GameModel):
    """returns yards gained"""
    def __init__(self, config):
        super().__init__(config["air_yard_cols"])
        self._load_model()
        self.model_name = 'AirYardsModel'

    def _load_model(self) -> None:
        model_path = "models/air_yards.pt"
        self.model = maskedModel(n_in=22, n_hidden=512, n_out=140).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))


    def predict(self, input: dict) -> int:
        
        features = self._fetch_model_input(input)
        x = torch.tensor(features).to(device)
        with torch.no_grad():
            preds = self.model(x.reshape(1, -1))[0]
            preds = torch.softmax(preds, 0)
        sample = int(torch.multinomial(preds, 1).item()) - 40
        return sample
    


class YacModel(GameModel):
    """returns yards gained"""
    def __init__(self, config):
        super().__init__(config["yac_cols"])
        self._load_model()
        self.model_name = 'YacModel'

    def _load_model(self) -> None:
        model_path = "models/yac.pt"
        self.model = maskedModelYac(n_in=26, n_hidden=256, n_out=140).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))


    def predict(self, input: dict) -> int:
        
        features = self._fetch_model_input(input)
        x = torch.tensor(features).to(device)
        with torch.no_grad():
            preds = self.model(x.reshape(1, -1))[0]
            preds = torch.softmax(preds, 0)
        sample = int(torch.multinomial(preds, 1).item()) - 40
        return sample
    
class OOBModel(GameModel):
    """returns boolean for if a play ended out of bounds or not"""
    def __init__(self, config):
        super().__init__(config["oob_cols"])
        self.play_decoder = config['play_decoding']
        self._load_model()
        self.model_name = 'OOBModel'

    def _load_model(self) -> None:
        self.model = joblib.load("models/oob_model.joblib")

    def predict(self, input: dict) -> int:
        features = self._fetch_model_input(input, ignore_missing=True)
        features
        preds = self.model.predict_proba([features])
        oob: int = np.random.choice(len(preds[0]), p=preds[0])
        return oob


class ClockModel(GameModel):
    """returns how long between one play starting, and the next play starting.
    No data to be able to do play clock runoff and play duration, so only one number."""
    def __init__(self, config):
        super().__init__(config["clock_cols"])
        self._load_model()
        self.play_encoding = config["play_encoding"]
        self.model_name = 'ClockModel'

    def _load_model(self) -> None:
        self.model = joblib.load("models/clock_model.joblib")

    def predict(self, input: dict) -> float:
        features = self._fetch_model_input(input, ignore_missing=True)
        duration = self.model.predict([features])[0]
        return duration
    

class FieldGoalModel(GameModel):
    """returns position and depth of rusher"""
    def __init__(self, config):
        super().__init__(config["fg_cols"])
        self._load_model()

    def _load_model(self) -> None:
        self.model = joblib.load("models/fg_model.joblib")

    def predict(self, input: dict) -> int:
        input['kick_distance'] = input['yardline_100'] + 10
        input['distance_sq'] = (input['kick_distance']) ** 2
        input: list = self._fetch_model_input(input)
        preds = self.model.predict_proba([input])
        result = np.random.choice(len(preds[0]), p=preds[0])
        return result
    