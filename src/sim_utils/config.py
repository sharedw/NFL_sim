import yaml

with open("models/feature_config.yaml", "r") as file:
	CONFIG = yaml.safe_load(file)