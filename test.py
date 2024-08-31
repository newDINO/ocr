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

from PIL import Image
from torchvision import transforms
to_tensor = transforms.ToTensor()

image = Image.open('data_gen/test.png').convert('RGB')
image = to_tensor(image).unsqueeze(0)

idx = torch.tensor([[0]], dtype=torch.long)

logits = model(idx, image)
print(logits)