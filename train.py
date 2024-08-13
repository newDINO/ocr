import torch
from torch import nn

# parameters
device = 'cpu'

block_size = 16
n_embed = 128
n_head = 8
n_layer = 1
dtype = torch.float32

# tokenizer
# all ascii characters, ' '(space) is for padding and end-of-text
vocab_size = 126 - 32 + 1
def encode(string):
    return [ord(char) - 32 for char in string]
def decode(idx):
    return ''.join([chr(index + 32) for index in idx])

# initiate dataset
texts = open('data_gen/data/random/texts.txt').read().split('\n')[:-1]
idx_data = []
for text in texts:
    idx = [0] + encode(text)
    while len(idx) <= block_size:
        idx.append(0)
    idx_data.append(torch.tensor(idx))
idx_data = torch.stack(idx_data)

from PIL import Image
from torchvision import transforms
to_tensor = transforms.ToTensor()
images = []
for i in range(len(texts)):
    image = Image.open(f"data_gen/data/random/imgs/{i}.png").convert("RGB")
    images.append(image)

# data loader

def load_data(images, idx, batch_size, device):
    random_pos = torch.randint(len(idx) - batch_size, (1,)).item()
    texts = idx[random_pos: random_pos + batch_size]
    x = texts[:, :-1].contiguous().to(device)
    y = texts[:, 1:].contiguous().to(device)
    imgs = images[random_pos: random_pos + batch_size]
    imgs_tensor = []
    for img in imgs:
        tensor = to_tensor(img)
        imgs_tensor.append(tensor)
    imgs_tensor = torch.stack(imgs_tensor).to(device)
    return x, imgs_tensor, y


from model import Model
model = Model(vocab_size, block_size, n_embed, n_layer, n_head).to(device, dtype)

x, img, y = load_data(images, idx_data, 4, device)
from code import interact
interact(local=locals())
logits, loss = model(x, img, y)
exit()

# model.compile()

optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=3e-4, eps=1e-7)

for _ in range(300):
    optimizer.zero_grad()
    x, img, y = load_data(images, idx_data, 32, device)
    _, loss = model(x, img, y)
    loss.backward()
    optimizer.step()
    print(loss.item())

# torch.save(model.state_dict(), "model.bin")