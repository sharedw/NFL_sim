import joblib
import torch
from sim_utils.modeling import WithDropout, masked_model
from abc import ABC, abstractmethod
import yaml
import numpy as np

with open("models/feature_config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


air_yards_path = "models/air_yards.pt"
AIR_YARDS_MODEL = WithDropout(n_in=28, n_out=119)
AIR_YARDS_MODEL.load_state_dict(torch.load(air_yards_path, weights_only=True))

yac_model_path = "models/yac.pt"
YAC_MODEL = WithDropout(n_in=29, n_out=125)
YAC_MODEL.load_state_dict(torch.load(yac_model_path, weights_only=True))

CLOCK_MODEL = joblib.load("models/clock_model.joblib")

CHOOSE_RUSHER_MODEL = joblib.load("models/choose_rusher.joblib")

CHOOSE_RECEIVER_MODEL = joblib.load("models/choose_receiver.joblib")

COMPLETE_PASS_MODEL = joblib.load("models/complete_pass.joblib")



class GameModel(ABC):
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = self._load_model()  # Force subclasses to implement
    
    @abstractmethod
    def _load_model(self):
        """Load the specific model - must be implemented by subclass"""
        pass
    
    @abstractmethod
    def predict(self, features):
        """Make prediction - must be implemented by subclass"""
        pass
    
    
    def _fetch_model_input(self, *feature_sources) -> list:
        combined = {}
        for source in feature_sources:
            if source is not None:
                combined.update(source)
        return [combined[col] for col in self.feature_cols]

class ChooseRusherModel(GameModel):
    """returns position and depth of rusher"""
    def __init__(self, config):
        super().__init__(config["choose_rusher_cols"])
        self._load_model()

    def _load_model(self) -> None:
        self.model = joblib.load("models/choose_rusher.joblib")

    def predict(self, *features: list[dict]) -> tuple[str,str]:
        features = self._fetch_model_input(*features)
        preds = self.model.predict_proba([features])
        rusher_idx = np.random.choice(len(preds[0]), p=preds[0])
        return rusher_idx
    

class RushYardsModel(GameModel):
    """returns yards gained"""
    def __init__(self, config):
        super().__init__(config["choose_rusher_cols"])
        self._load_model()

    def _load_model(self) -> None:
        model_path = "models/run_yards_gained.pt"
        self.model = masked_model(n_in=7, n_hidden=256, n_out=130).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))


    def predict(self, *features: list[dict]) -> int:
        
        features = self._fetch_model_input(*features)
        x = torch.tensor(features).to(device)
        with torch.no_grad():
            preds = self.model(x.reshape(1, -1))[0]
            preds = torch.softmax(preds, 0)
        sample = (torch.multinomial(preds, 1)).item() - 30
        return sample

