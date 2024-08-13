import torch
import math
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embed,
        n_layer,
        n_head,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(block_size, n_embed)

        self.img_encoder = ConvEncoder(n_embed=n_embed)

        self.cross = LayerWithCrossAttn(n_embed=n_embed, n_head=n_head)

        # self.h = Layer(n_embed=n_embed, n_head=n_head)

        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.block_size = block_size

        # self.lm_head.weight = self.wte.weight

        # self.apply(self._init_weights)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)


    def forward(self, idx, image, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb

        y = self.img_encoder(image)
        x = self.cross(x, y)
        # x = self.h(x)

        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class Layer(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed=n_embed, n_head=n_head)

        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = Mlp(n_embed=n_embed)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
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
        self.attn = CausalSelfAttention(n_embed=n_embed, n_head=n_head)

        self.ln21 = nn.LayerNorm(n_embed)
        self.ln22 = nn.LayerNorm(n_embed)
        self.cross_attn = CrossAttention(n_embed=n_embed, n_head=n_head)

        self.ln3 = nn.LayerNorm(n_embed)
        self.mlp = Mlp(n_embed=n_embed)
    def forward(self, x, y):
        x = x + self.attn(self.ln1(x))
        x = x + self.cross_attn(self.ln21(x), self.ln22(y))
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

class CausalSelfAttention(nn.Module):
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

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head
    ):
        super().__init__()
        self.c_atten = nn.Linear(n_embed, n_embed)
        self.y_atten = nn.Linear(n_embed, 2 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
    def forward(self, x, y):
        B, Tx, C = x.size()
        _, Ty, _ = y.size()
        q = self.c_atten(x)
        kv = self.y_atten(y)
        k, v = kv.split(self.n_embed, dim=2)

        q = q.view(B, Tx, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, Ty, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, Ty, self.n_head, C // self.n_head).transpose(1, 2)

        result = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
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
        # 3x128x256
        self.conv1 = nn.Conv2d(3, n_embed // 4, (4, 4), 4)
        self.relu1 = nn.ReLU()
        # _x32x64
        self.conv2 = nn.Conv2d(n_embed // 4, n_embed // 2, (4, 4), 4)
        self.relu2 = nn.ReLU()
        # _x8x16
        self.conv3 = nn.Conv2d(n_embed // 2, n_embed, (2, 2), 2)
        self.relu3 = nn.ReLU()
        # _x4x8
        self.wpe = nn.Embedding(32, n_embed)
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        B, C, H, W = x.size()
        pos = torch.arange(0, 32, dtype=torch.long, device=x.device)
        pos_emb = self.wpe(pos)
        x  = x.view(B, C, -1).transpose(1, 2) + pos_emb

        return x