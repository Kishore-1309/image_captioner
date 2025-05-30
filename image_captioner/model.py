import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class TransformerBridge(nn.Module):
    def __init__(self, image_feature_dim=512, gpt2_hidden_dim=768, 
                 num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.image_to_hidden = nn.Linear(image_feature_dim, gpt2_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gpt2_hidden_dim, 
            nhead=num_heads,
            dim_feedforward=2048, 
            batch_first=True,
            #dropout=dropout,
            activation='gelu'
        )
        self.image_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, image_features):
        x = self.image_to_hidden(image_features).unsqueeze(1)
        return self.image_encoder(x)

class CLIPGPT2CaptionModel(nn.Module):
    def __init__(self, bridge, gpt2):
        super().__init__()
        self.bridge = bridge
        self.gpt2 = gpt2

    def forward(self, input_ids, attention_mask, image_features, labels=None):
        image_embeds = self.bridge(image_features)
        return self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.shape[:2]).to(image_embeds.device),
            labels=labels
        )