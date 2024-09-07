import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest

'''
This ResNet class is intended to be used as the smallest unit of the block class
'''
class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32):
        super().__init__()
        num_groups = min(in_channels, 4)
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.GroupNorm(num_groups,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.network(x)
        return torch.add(out,self.residual_layer(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.n_heads < config.emb_dim:
            self.n_heads = 1
        else:
            self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim

        self.q = nn.Linear(config.emb_dim, config.emb_dim)
        self.k = nn.Linear(config.emb_dim, config.emb_dim)
        self.v = nn.Linear(config.emb_dim, config.emb_dim)

        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                     .view(1, 1, config.seq_len, config.seq_len))
        self.ff = nn.Linear(config.emb_dim, config.emb_dim)
            
    def forward(self,q,k,v):
        B, T, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        qk = qk.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v
        qkv = qkv.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        out = self.ff(qkv)
        return out

class Transformer(nn.Module):
    def __init__(self, config, use_mask = True):
        super().__init__()
        self.use_mask = use_mask
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.mha1 = nn.MultiheadAttention(config.emb_dim, config.num_heads, batch_first=True)
        self.mha = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * 10),
            nn.GELU(),
            nn.Linear(config.emb_dim * 10, config.emb_dim)
        )       

    def forward(self, x):
        x_norm = self.ln1(x)
        if not self.use_mask:
            attn_output,_ = self.mha1(x_norm, x_norm, x_norm)
        else:
            attn_output = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Feed Forward + Add & Norm
        x_norm = self.ln2(x)
        ff_output = self.ff(x_norm)
        x = x + ff_output
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 6, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x_size = x.shape[-1]
        batch_size = x.shape[0]
        # Using reshape instead of view
        x = x.reshape(batch_size, self.channels, -1).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.dropout(attention_value)
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).reshape(batch_size, self.channels, x_size, x_size)
    
class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.use_mask = config.use_mask
        self.mha1 = nn.MultiheadAttention(config.emb_dim, config.n_heads, batch_first=True)
        self.mha = CausalSelfAttention(config)
        self.ln0 = nn.LayerNorm(config.emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * 10),
            nn.GELU(),
            nn.Linear(config.emb_dim * 10, config.emb_dim)
        )
        self.ln1 = nn.LayerNorm(config.emb_dim)

    def forward(self, x):
        skip = x
        if not self.use_mask:
            x, _ = self.mha1(x, x, x)
        else:
            x = self.mha(x,x,x)
        
        x = self.ln0(x + skip)
        skip = x
        x = self.ff(x)
        x = self.ln1(x + skip)
        return x

        
'''
Unit testing class
'''
class TestResNet(unittest.TestCase):
    def test_forward(self):
        '''  
        model = ResNet(in_channels=64,out_channels = 16)
        input_tensor = torch.randn(1, 64, 64, 64)  # Example input with shape (batch_size, channels, height, width)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape, (1, 16, 64, 64))  # Adjust the expected shape based on your model architecture
        
        

        model = SelfAttention(16,64)
        input_tensor = torch.randn(3, 16, 64, 64)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape,(3,16,64,64))
        '''
        


        
if __name__ == '__main__':
    unittest.main()