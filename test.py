import torch

from model import Model

device = 'cuda'
block_size = 64
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

def cal_n_param(model):
    return sum(p.numel() for p in model.parameters())

print(cal_n_param(model))