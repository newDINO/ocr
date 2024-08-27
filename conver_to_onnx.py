import torch
import torch
import math
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        img_block_size,
        n_embed,
        n_layer,
        n_head,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(block_size, n_embed)

        self.img_encoder = ConvEncoder(n_embed=n_embed, block_size=img_block_size)
        
        self.encoder = nn.ModuleList([Layer(n_embed, n_head) for _ in range(n_layer)])
        self.decoder = nn.ModuleList([LayerWithCrossAttn(n_embed, n_head) for _ in range(n_layer)])

        # self.h = Layer(n_embed=n_embed, n_head=n_head)

        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.block_size = block_size

        # self.lm_head.weight = self.wte.weight
        # self.apply(self._init_weights)



    def forward(self, idx, image):
        # idx is of shape (B, T)
        B, T = idx.size()
        # assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb

        y = self.img_encoder(image)

        # for (encoder_layer, decoder_layer) in zip(self.encoder, self.decoder):
        #     y = encoder_layer(y)
        #     x = decoder_layer(x, y)
        for i in range(len(self.encoder)):
            y = self.encoder[i](y)
            x = self.decoder[i](x, y)

        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # return logits

        index = logits[:, -1, :].argmax(-1)
        return index


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

        result = F.scaled_dot_product_attention(q, k, v) # flash attention
        result = result.transpose(1, 2).contiguous().view(B, Tx, C) # re-assemble all head outputs side by side
        # output projection
        result = self.c_proj(result)
        return result


class ConvEncoder(nn.Module):
    def __init__(
        self,
        n_embed,
        block_size
    ):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.proj = nn.Conv2d(in_channels=64, out_channels=n_embed, kernel_size=4, stride=4)

        self.wpe = nn.Embedding(block_size, n_embed)
        self.block_size = block_size
    
    def _init_weight_from_cnn(self, cnn):
        self.conv1.weight = cnn.conv1.weight
        self.conv2.weight = cnn.conv2.weight
        self.conv3.weight = cnn.conv3.weight
    
    def _freeze_cnn_layers(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Convolution -> Activation -> Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 64, 128)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (32, 32, 64)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (64, 16, 32)
        x = F.relu(self.proj(x)) # Output: (n_embed, 4, 8)
        
        B, C, H, W = x.size()
        x = x.view(B, C, -1).transpose(1, 2)
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        pos_emb = self.wpe(pos)
        x  = x + pos_emb

        return x

device = 'cuda'
block_size = 16
img_block_size = 32
n_embed = 512
n_head = 8
n_layer = 2
dtype = torch.float32
vocab_size = 126 - 32 + 1

model = Model(
    vocab_size=vocab_size,
    block_size=block_size,
    img_block_size=img_block_size,
    n_embed=n_embed,
    n_layer=2,
    n_head=n_head,
)

model.load_state_dict(torch.load("models/2layersMath3.bin", map_location='cpu'))

dummy_image_input = torch.randn(1, 3, 128, 256)
dummy_idx_input = torch.randint(0, vocab_size, (1, block_size), dtype=torch.long)

# model = torch.jit.script(model)

torch.onnx.export(
    model,
    (dummy_idx_input, dummy_image_input),
    "front_end/model.onnx",
    # verbose=True,
    input_names=['idx', 'image'],
    output_names=['output'],
    dynamic_axes={
        "idx": {1: "block_size"},
    }
)