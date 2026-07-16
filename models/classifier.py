# import torch
# import torch.nn as nn

# class LSPAGatedNetwork(nn.Module):
#     def __init__(self, num_layers=4, hidden_dim=2048, bottleneck_dim=512):
#         super(LSPAGatedNetwork, self).__init__()
        
#         # FC1: Dimensionality Reduction
#         self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(p=0.3)
        
#         # FC2: Layer Importance Scoring
#         self.fc2 = nn.Linear(bottleneck_dim, 1)
        
#         # FC3: Cross-Layer Fusion
#         self.fc3 = nn.Linear(num_layers, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, combined_features):
#         x = self.fc1(combined_features)
#         x = self.gelu(x)
#         x = self.dropout(x)
        
#         x = self.fc2(x)
#         x = x.squeeze(-1)
        
#         logits = self.fc3(x)
#         return self.sigmoid(logits).squeeze(-1) # Squeeze to match label shape [Batch]

import torch
import torch.nn as nn

class LSPAAttentionGatedNetwork(nn.Module):
    def __init__(self, num_layers=4, feature_dim=2048, hidden_dim=512):
        super(LSPAAttentionGatedNetwork, self).__init__()
        
        # Project each 2048-dim layer down to a manageable space
        self.layer_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
        # Multi-Head Attention layer to look across the 4 layers
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input shape expected: [batch_size, 4, 2048]
        batch_size, num_layers, feat_dim = x.shape
        
        # Step 1: Project layers individually -> [batch_size, 4, 512]
        projected = torch.stack([self.layer_projection(x[:, i, :]) for i in range(num_layers)], dim=1)
        
        # Step 2: Pass through Self-Attention -> [batch_size, 4, 512]
        attn_output, _ = self.multihead_attention(projected, projected, projected)
        
        # Step 3: Flatten the cross-attentive features -> [batch_size, 4 * 512]
        flattened = attn_output.reshape(batch_size, -1)
        
        # Step 4: Classify
        return self.classifier(flattened).squeeze(-1)