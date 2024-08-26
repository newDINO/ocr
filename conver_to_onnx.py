import torch

from model import Model

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

model.load_state_dict(torch.load("models/2layersGen5.bin", map_location='cpu'))

dummy_image_input = torch.randn(1, 3, 128, 256)
dummy_idx_input = torch.randint(0, vocab_size, (1, block_size), dtype=torch.long)

model = torch.jit.script(model)

torch.onnx.export(
    model,
    (dummy_idx_input, dummy_image_input),
    "models/model.onnx",
    verbose=True,
    input_names=['idx', 'image'],
    output_names=['output'],
    dynamic_axes={
        "idx": {1: "block_size"},
        "output": {1: "block_size"},
    }
)