import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from clip_modules import ResNet,Block,SelfAttention,Transformer
from einops.layers.torch import Rearrange

class CLIPConfig:
    def __init__(self, width=64, latent_dim=128, n_layers=8, emb_dim=128, 
                 vocab_len=1024, seq_len=512, n_heads=8, dropout=0.2, 
                 image_size = 128, initial_temperature = 1, use_mask = True):
        self.width = width
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.vocab_len = vocab_len
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.dropout = dropout
        self.image_size = image_size
        self.initial_temperature = initial_temperature
        self.use_mask = use_mask


# CNNImageEncoder Class
class CNNImageEncoder(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, config.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.width, config.width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.width * 2, config.width * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(config.width * 3, config.width)
        )
        self.ff = nn.Sequential(
            nn.LayerNorm([config.width * (config.image_size//4)**2]),
            nn.Linear(config.width * (config.image_size//4)**2, config.width * 10),
            nn.GELU(),
            nn.Linear(config.width * 10, config.latent_dim)
        )

    def forward(self, x):
        x = self.net(x)
        
        x = torch.flatten(x, 1)  # Flatten before passing to the feed-forward network
        
        latent = self.ff(x)
        return latent

# TextEncoder Class
class TextEncoder(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.tok_emb = nn.Embedding(config.vocab_len, config.emb_dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.emb_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm([config.emb_dim]),
            nn.Linear(config.emb_dim, config.emb_dim * 10),
            nn.GELU(),
            nn.Linear(config.emb_dim * 10, config.latent_dim)
        )
    
    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)
        tok_emb = self.tok_emb(x)
        emb = tok_emb + pos_emb.unsqueeze(0)

        for block in self.blocks:
            emb = block(emb)
        
        latent = self.ff(emb.mean(dim=1))  # Apply mean pooling before final ff
        return latent

class ViT(nn.Module):
    def __init__(self, in_channels = 3, p = 2, dim = 128, depth = 8):
        super(ViT, self).__init__()
        patch_dim = in_channels * p * p
        self.num_patches = (128 // p) * (128 // p)
        in_dim = (128//2)**2 * 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.att = nn.ModuleList([])
        for _ in range(depth):
            self.att.append(Transformer(dim))
        self.pos_emb = nn.Embedding(self.num_patches,dim)
        self.proj = nn.Linear(dim,2)
        self.out = nn.Linear(in_dim,256)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        pos = torch.arange(0, self.num_patches, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)
        x += pos

        for layer in self.att:
            x = layer(x)

        x = self.proj(x)
        x = torch.flatten(x,1)
        x = self.out(x)
        return x

        
# CLIP Class
class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.img_encoder = CNNImageEncoder(config)
        self.txt_encoder = TextEncoder(config)
        self.temperature = nn.Parameter(torch.tensor(config.initial_temperature, dtype=torch.float32))
    
    def forward(self, img, tkns):
        image_features = self.img_encoder(img)
        text_features = self.txt_encoder(tkns)

        # Normalize the image and text features to have unit length
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        temperature_exp = self.temperature.exp()

        # Compute the cosine similarity matrix
        logits_per_image = torch.matmul(image_features, text_features.T) * temperature_exp
        logits_per_text = torch.matmul(text_features, image_features.T) * temperature_exp
        
        img_probs = F.softmax(logits_per_image)
        txt_probs = F.softmax(logits_per_text)
        return img_probs,txt_probs
    
    def calculate_loss(self,img, tkns):
        image_features = self.img_encoder(img)
        
        text_features = self.txt_encoder(tkns)

        # Normalize the image and text features to have unit length
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        temperature_exp = self.temperature

        # Compute the cosine similarity matrix
        logits_per_image = torch.matmul(image_features, text_features.T) / temperature_exp
        logits_per_text = torch.matmul(text_features, image_features.T) / temperature_exp

        # Labels for the contrastive loss: diagonal elements should be the highest
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        # Compute cross-entropy loss in both directions
        loss_image_to_text = F.cross_entropy(logits_per_image, labels)
        loss_text_to_image = F.cross_entropy(logits_per_text, labels)

        # Total loss is the average of the two
        loss = (loss_image_to_text + loss_text_to_image) / 2
        return loss