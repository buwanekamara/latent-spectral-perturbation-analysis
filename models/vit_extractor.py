import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- Added for dynamic interpolation
from transformers import CLIPVisionModel

class FrozenViTExtractor(nn.Module):
    def __init__(self, target_layers):
        super(FrozenViTExtractor, self).__init__()
        print("Initializing Frozen CLIP ViT-L/14...")
        self.vit = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            output_hidden_states=True
        )
        self.target_layers = target_layers
        
        # Freeze the model completely
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, orig_pixel_values, stressed_pixel_values):
        # Fix: Downsample from 256x256 to 224x224 for CLIP compliance
        # Using align_corners=False to safely mirror standard bilinear resizing
        orig_224 = F.interpolate(orig_pixel_values, size=(224, 224), mode='bilinear', align_corners=False)
        stressed_224 = F.interpolate(stressed_pixel_values, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            orig_outputs = self.vit(orig_224)
            stressed_outputs = self.vit(stressed_224)

        orig_hidden = orig_outputs.hidden_states
        stressed_hidden = stressed_outputs.hidden_states

        combined_layers = []
        for layer_idx in self.target_layers:
            # Extract [CLS] token at index 0
            orig_cls = orig_hidden[layer_idx][:, 0, :]
            stressed_cls = stressed_hidden[layer_idx][:, 0, :]
            
            # Calculate Delta
            delta = torch.abs(orig_cls - stressed_cls)
            
            # Context Concatenation
            combined = torch.cat([orig_cls, delta], dim=-1)
            combined_layers.append(combined)

        # Shape: [Batch, len(target_layers), 2048]
        return torch.stack(combined_layers, dim=1)