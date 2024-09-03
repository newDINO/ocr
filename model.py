import torch
import math
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed,
        n_layer,
        n_head,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embed)

        self.img_encoder = ConvEncoder(n_embed=n_embed)
        
        self.encoder = nn.ModuleList([Layer(n_embed, n_head) for _ in range(n_layer)])
        self.decoder = nn.ModuleList([LayerWithCrossAttn(n_embed, n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx, image):
        # idx is of shape (B, T)
        B, T = idx.size()
        # assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        y = self.img_encoder(image)

        idx_pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # shape (1, T)
        img_pos = torch.arange(0, y.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb

        for i in range(len(self.encoder)):
            y = self.encoder[i](y, img_pos)
            x = self.decoder[i](x, y, idx_pos, img_pos)

        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


class Layer(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = RopeSelfAttention(n_embed=n_embed, n_head=n_head)

        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = Mlp(n_embed=n_embed)
    def forward(self, x, position_ids):
        x = x + self.attn(self.ln1(x), position_ids)
        x = x + self.mlp(self.ln2(x))
        return x


class LayerWithCrossAttn(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = RopeSelfAttention(n_embed=n_embed, n_head=n_head)

        self.ln21 = nn.LayerNorm(n_embed)
        self.ln22 = nn.LayerNorm(n_embed)
        self.cross_attn = RopeCrossAttention(n_embed=n_embed, n_head=n_head)

        self.ln3 = nn.LayerNorm(n_embed)
        self.mlp = Mlp(n_embed=n_embed)
    def forward(self, x, y, x_pos, y_pos):
        x = x + self.attn(self.ln1(x), x_pos)
        x = x + self.cross_attn(self.ln21(x), self.ln22(y), x_pos, y_pos)
        x = x + self.mlp(self.ln3(x))
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        n_embed,
    ):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * n_embed, n_embed)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [1, seq_len]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # [1, dim / 2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # [1, 1, seq_len]

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [1, seq_len, dim / 2]

        emb = torch.cat((freqs, freqs), dim=-1) # [1, seq_len, dim]
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    # q: [batch_size, heads, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class RopeSelfAttention(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head,
    ):
        super().__init__()
        assert n_embed % n_head == 0
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, position_ids):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # [B, n_head, T, head_size]

        cos, sin = self.rotary_emb(x, position_ids)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(q, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class RopeCrossAttention(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head,
    ):
        super().__init__()
        assert n_embed % n_head == 0
        self.c_attn = nn.Linear(n_embed, n_embed)
        self.y_attn = nn.Linear(n_embed, 2 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, y, x_pos, y_pos):
        B, Tx, C = x.size()
        _, Ty, _ = y.size()
        q = self.c_attn(x)
        kv = self.y_attn(y)
        k, v = kv.split(self.n_embed, dim=2)

        q = q.view(B, Tx, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, Ty, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, Ty, self.n_head, self.head_dim).transpose(1, 2)

        cos_q, sin_q = self.rotary_emb(x, x_pos)
        cos_k, sin_k = self.rotary_emb(y, y_pos)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)

        result = F.scaled_dot_product_attention(q, k, v) # flash attention
        result = result.transpose(1, 2).contiguous().view(B, Tx, C) # re-assemble all head outputs side by side
        # output projection
        result = self.c_proj(result)
        return result



class ConvEncoder(nn.Module):
    def __init__(
        self,
        n_embed,
    ):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.proj = nn.Conv2d(in_channels=64, out_channels=n_embed, kernel_size=4, stride=4)
    
    def forward(self, x):
        # Convolution -> Activation -> Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 64, 128)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (32, 32, 64)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (64, 16, 32)
        x = F.relu(self.proj(x)) # Output: (n_embed, 4, 8)
        
        B, C, H, W = x.size()
        x = x.view(B, C, -1).transpose(1, 2)

        return x