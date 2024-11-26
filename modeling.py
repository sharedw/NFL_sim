import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
#11, 109
#23, 119
#24, 125
class ThreeLayerClassifier(nn.Module):
    def __init__(self, n_in=11, n_out=109, n_hidden=50):
        super(ThreeLayerClassifier, self).__init__()
        self.hidden1 = nn.Linear(n_in, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden + 1, n_out)  # +1 for the additional feature
    def forward(self, x):
        # Extract the feature to pass to the final layer (e.g., the first feature)
        feature_to_pass = x[:, 0].unsqueeze(1)  # Assuming you want the first feature, shape [batch_size, 1]
        
        # Pass through the first two layers
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        # Concatenate the extracted feature with the output of the hidden layers
        x = torch.cat((x, feature_to_pass), dim=1)  # Concatenate along the feature axis (dim=1)
        
        # Pass through the final layer
        x = self.output_layer(x)
        return x


def create_model(df, x_cols, y_col, colsample_bytree=0.5):
    data = df.loc[~(df[y_col].isna())].copy(deep=True)
    X, y = data[x_cols], data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBClassifier(eval_metric="mlogloss", colsample_bytree=colsample_bytree)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    display(cm) # type: ignore
    return model


def create_reg_model(df, x_cols, y_col):
    data = df.loc[~(df[y_col].isna())].copy(deep=True)
    X, y = data[x_cols], data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=100, colsample_bytree=0.5
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