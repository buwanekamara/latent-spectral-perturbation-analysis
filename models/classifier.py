import torch
import torch.nn as nn

class LSPAGatedNetwork(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=2048, bottleneck_dim=512):
        super(LSPAGatedNetwork, self).__init__()
        
        # FC1: Dimensionality Reduction
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)
        
        # FC2: Layer Importance Scoring
        self.fc2 = nn.Linear(bottleneck_dim, 1)
        
        # FC3: Cross-Layer Fusion
        self.fc3 = nn.Linear(num_layers, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, combined_features):
        x = self.fc1(combined_features)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.squeeze(-1)
        
        logits = self.fc3(x)
        return self.sigmoid(logits).squeeze(-1) # Squeeze to match label shape [Batch]