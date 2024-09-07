import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import math


class Config:
    def __init__(self, num_heads = 8 , emb_dim = 128, use_mask=True, seq_len=256, num_layers = 8, vocab_len = 512):
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.use_mask = use_mask
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.vocab_len = vocab_len

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.num_heads < config.emb_dim:
            self.n_heads = 1
        else:
            self.n_heads = config.num_heads
        self.emb_dim = config.emb_dim

        self.q = nn.Linear(config.emb_dim, config.emb_dim)
        self.k = nn.Linear(config.emb_dim, config.emb_dim)
        self.v = nn.Linear(config.emb_dim, config.emb_dim)

        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                     .view(1, 1, config.seq_len, config.seq_len))
        self.ff = nn.Linear(config.emb_dim, config.emb_dim),
            
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
        
        return qkv



class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.emb_dim < config.num_heads:
            config.num_heads = 1
        self.y_ln = nn.LayerNorm(config.emb_dim)
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.mha1 = nn.MultiheadAttention(config.emb_dim, config.num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.mha2 = nn.MultiheadAttention(config.emb_dim, config.num_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(config.emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * 10),
            nn.GELU(),
            nn.Linear(config.emb_dim * 10, config.emb_dim),
        )
        self.seq_len = config.seq_len
        self.use_mask = config.use_mask

    def forward(self, x, y = None):
        x_norm = self.ln1(x)
        attn_output = self.mha1(x_norm, x_norm, x_norm)[0]
        x = x + attn_output

        # Feed Forward + Add & Norm
        x_norm = self.ln2(x)
        
        if y is not None:
            y = self.y_ln(y)
            attn_output = self.mha2(x_norm, y, y)[0]
            x = x + attn_output
            x_norm = self.ln3(x)
        
        ff_output = self.ff(x_norm)
        x = x + ff_output

        return x

class Muse(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.first_block = Block(config)
        config.use_mask = False  # Set use_mask to False for subsequent blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers - 1)])
        self.ln = nn.LayerNorm(config.emb_dim)
        self.head = nn.Linear(config.emb_dim, config.vocab_len)
        self.tok_emb = nn.Embedding(config.vocab_len + 1, config.emb_dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.emb_dim)
        self.clip_proj = nn.Linear(256,config.emb_dim)
        self.emb_dim = config.emb_dim

    def forward(self, x, c = None, y = None):
        B, T = x.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)

        tok_emb = self.tok_emb(x)
        
        emb = tok_emb + pos_emb.unsqueeze(0)
        
        emb = self.first_block(emb)
        
        if c is not None:
            c = self.clip_proj(c)
            c = c.unsqueeze(1)
            #c = c.repeat(1,1,self.emb_dim)
            for block in self.blocks:
                emb = block(emb,c)

        else:
            for block in self.blocks:
                emb = block(emb)
        
        emb = self.ln(emb)
        logits = self.head(emb)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
    

# Unit test
class TestModels(unittest.TestCase):
    def test_block(self):
        config = Config()
        batch_size = 2
        x = torch.randn(batch_size, config.seq_len, config.emb_dim)

        block = Block(config)
        output = block(x)
        self.assertEqual(output.shape, (batch_size, config.seq_len, config.emb_dim))

    def test_Muse(self):
        config = Config(
            emb_dim=8
        )
        batch_size = 2
        x = torch.randint(0, config.vocab_len, (batch_size, config.seq_len))
        c = torch.randint(0, config.vocab_len, (batch_size, 256))
        c = c.float()

        muse = Muse(config)
        logits, loss = muse(x,c)
        self.assertEqual(logits.shape, (batch_size, config.seq_len, config.vocab_len))

        y = torch.randint(0, config.vocab_len, (batch_size, config.seq_len))
        logits, loss = muse(x,c,y)
        print(loss)
        self.assertIsNotNone(loss)

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
