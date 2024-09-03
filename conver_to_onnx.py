import torch
import torch
import math
from torch import nn
import torch.nn.functional as F
from model import Model

class RuntimeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, idx, img):
        logits = self.model(idx, img)
        return logits[:, -1, :].argmax(-1)
        

device = 'cuda'
dummy_block_size = 48
n_embed = 512
n_head = 8
n_layer = 2
dtype = torch.float32
from tokenizer import vocab_size

model = Model(
    vocab_size=vocab_size,
    n_embed=n_embed,
    n_layer=n_layer,
    n_head=n_head,
)

model.load_state_dict(torch.load("models/latex19.bin", map_location='cpu'))

runtime_model = RuntimeModel(model)

dummy_image_input = torch.randn(1, 3, 128, 256)
dummy_idx_input = torch.randint(0, vocab_size, (1, dummy_block_size), dtype=torch.long)


torch.onnx.export(
    runtime_model,
    (dummy_idx_input, dummy_image_input),
    "front_end/model.onnx",
    # verbose=True,
    input_names=['idx', 'image'],
    output_names=['output'],
    dynamic_axes={
        "idx": {1: "block_size"},
    }
)