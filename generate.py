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


from PIL import Image
from torchvision import transforms
to_tensor = transforms.ToTensor()

image = Image.open('data_gen/data/texts/l1/0.png').convert('RGB')
image = to_tensor(image).unsqueeze(0)


def generate(model, image, eos, max_len):
    idx = torch.tensor([[0]], dtype=torch.long)
