import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)


def adjacent_label_smoothing(labels, yardline, num_classes=130, smoothing=0.1, k=3, yard_shift=40):
	"""
	labels: [batch]
	yardline: [batch]
	k: number of bins to smooth on each side
	"""
	batch_size = labels.size(0)
	device = labels.device
	
	smoothed = torch.zeros(batch_size, num_classes, device=device)
	idx = torch.arange(batch_size, device=device)

	weight = np.array([x for x in range(k+1, 1, -1)])
	weight = 0.5 * weight / sum(weight)

	
	# main label
	smoothed[idx, labels] = 1.0 - smoothing

	# left and right neighbors
	for offset, w  in zip(range(1, k+1), weight):
		# left
		left_idx = labels - offset
		valid_left = left_idx >= 0
		smoothed[idx[valid_left], left_idx[valid_left]] += w

		# right
		right_idx = labels + offset
		valid_right = right_idx < num_classes
		smoothed[idx[valid_right], right_idx[valid_right]] += w

		# edge correction
		smoothed[idx[~valid_left], labels[~valid_left]] += w
		smoothed[idx[~valid_right], labels[~valid_right]] += w

	# mask impossible yards
	mask = torch.arange(num_classes, device=device).expand(batch_size, -1) > (yardline + yard_shift).unsqueeze(1)
	smoothed[mask] = 0

	return smoothed

class FocalLoss(nn.Module):
	def __init__(self, gamma=2.0):
		super().__init__()
		self.gamma = gamma
	
	def forward(self, logits, targets):
		ce_loss = F.cross_entropy(logits, targets, reduction='none')
		pt = torch.exp(-ce_loss)
		focal_loss = ((1 - pt) ** self.gamma) * ce_loss
		return focal_loss.mean()
	
class FocalLossSoft(nn.Module):
	def __init__(self, gamma=2.0):
		super().__init__()
		self.gamma = gamma

	def forward(self, logits, targets):  # targets: [B, num_classes]
		log_probs = F.log_softmax(logits, dim=1)
		probs = log_probs.exp()
		loss = -((1 - probs) ** self.gamma) * targets * log_probs
		return loss.sum(dim=1).mean()


class ResBlock(nn.Module):
	def __init__(self, n_hidden):
		super().__init__()
		self.layer = nn.Sequential(
			nn.LayerNorm(n_hidden),
			nn.Linear(n_hidden, n_hidden),
			nn.ReLU()
		)
	
	def forward(self, x):
		return x + self.layer(x)



class maskedModel(nn.Module):
	def __init__(self, n_in=11, n_out=140, n_hidden=512):
		super(maskedModel, self).__init__()
		self.n_out=n_out
		self.main_layers = nn.Sequential(
		nn.Linear(n_in, n_hidden),nn.ReLU(),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		#ResBlock(n_hidden),
		#ResBlock(n_hidden),
		#ResBlock(n_hidden),
		)
		self.output_layer  = nn.Linear(n_hidden, n_out)
		self.td_head = nn.Sequential(nn.Linear(n_in, n_hidden),nn.ReLU(),
		ResBlock(n_hidden), nn.Linear(n_hidden,1))
		
	def forward(self, x):
		# Extract the feature to pass to the final layer (e.g., the first feature)
		yardline = x[:, 0].unsqueeze(1)  # Assuming you want the first feature, shape [batch_size, 1]
		#td_bin = (yardline + 40).long().squeeze(1)
		#td_logits = self.td_head(x)
		x = self.main_layers(x)
		logits = self.output_layer(x)
		#batch_indices = torch.arange(logits.size(0), device=logits.device)
		#logits[batch_indices, td_bin] = logits[batch_indices, td_bin] + td_logits.squeeze(1)
		
		yard_values = torch.arange(-40, self.n_out - 40, device=x.device).float().unsqueeze(0)
		mask = (yard_values <= yardline).float()
		logits = logits + (mask - 1) * 1e9

		return logits
	
class maskedModelYac(nn.Module):
	def __init__(self, n_in=11, n_out=140, n_hidden=512, offset=40):
		super(maskedModelYac, self).__init__()
		self.n_out=n_out
		self.offset=40
		self.main_layers = nn.Sequential(
		nn.Linear(n_in, n_hidden),nn.ReLU(),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		ResBlock(n_hidden),
		)
		self.output_layer  = nn.Linear(n_hidden, n_out)
		self.td_head = nn.Sequential(ResBlock(n_hidden), nn.Linear(n_hidden,1))
		
	def forward(self, x):
		logits=0 # need for weird cuda error?
		# Extract the feature to pass to the final layer (e.g., the first feature)
		yardline = x[:, 0].unsqueeze(1)  # Assuming you want the first feature, shape [batch_size, 1]
		air_yards = x[:, 2].unsqueeze(1)
		td_bin = (yardline + self.offset).long().squeeze(1)
		x = self.main_layers(x)
		td_logits = self.td_head(x)
		logits = self.output_layer(x)
		batch_indices = torch.arange(logits.size(0), device=logits.device)
		logits[batch_indices, td_bin] = logits[batch_indices, td_bin] + td_logits.squeeze(1)
		
		yard_values = torch.arange(-self.offset, self.n_out - self.offset, device=x.device).float().unsqueeze(0)
		mask = (yard_values <= (yardline - air_yards)).float()
		logits = logits + (mask - 1) * 1e3

		return logits



def create_model(df, x_cols, y_col, colsample_bytree=0.5):
    data = df.loc[~(df[y_col].isna())].copy(deep=True)
    X, y = data[x_cols], data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBClassifier(eval_metric="mlogloss", colsample_bytree=colsample_bytree, missing=np.nan)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return model


def create_reg_model(df, x_cols, y_col):
    data = df.loc[~(df[y_col].isna())].copy(deep=True)
    X, y = data[x_cols], data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        colsample_bytree=0.5,
        enable_categorical=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return model


def update_config(new_config):
    yaml_path = "models/feature_config.yaml"
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)  # Load the current content of the YAML file

    # Update the data with the provided updates
    data.update(new_config)

    with open(yaml_path, "w") as file:
        yaml.safe_dump(data, file)
