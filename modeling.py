import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeLayerClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=110):
        super(ThreeLayerClassifier, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size + 1, output_size)  # +1 for the additional feature
    def forward(self, x):
        # Extract the feature to pass to the final layer (e.g., the first feature)
        feature_to_pass = x[:, 0].unsqueeze(1)  # Assuming you want the first feature, shape [batch_size, 1]
        
        # Pass through the first two layers
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        
        # Concatenate the extracted feature with the output of the hidden layers
        x = torch.cat((x, feature_to_pass), dim=1)  # Concatenate along the feature axis (dim=1)
        
        # Pass through the final layer
        x = self.output_layer(x)
        return x