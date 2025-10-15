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
from sim_utils.Event import EventLog
from time import time






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

