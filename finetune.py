import torch
import torch.nn.functional as F
from torchvision import transforms
from model import Model
from PIL import Image
from tokenizer import encode, special_token_ids, vocab_size

n_embed = 512
n_head = 8
n_layer = 2
model = Model(vocab_size=vocab_size, n_embed=n_embed, n_head=n_head, n_layer=n_layer)
model.load_state_dict(torch.load("models/hand_math19.bin", map_location="cpu"))

def load_data():
    imgs = []
    idx = []
    begin_id = special_token_ids['<begin>']
    end_id = special_token_ids['<eos>']
    
    texts = open("data_gen/data/human/text.txt").read()
    for line in texts.split("\n"):
        if line == "":
            continue
        ids = [begin_id] + encode(line) + [end_id]
        idx.append(ids)
    
    for i in range(len(idx)):
        img = Image.open(f"data_gen/data/human/{i + 1}.png").convert('RGB')
        imgs.append(img)

    return imgs, idx

to_tensor = transforms.ToTensor()

def get_data(idx, imgs, i):
    idx = torch.tensor(idx[i], dtype=torch.long)
    x = idx[:-1].unsqueeze(0)
    y = idx[1:].unsqueeze(0)
    img = to_tensor(imgs[i]).unsqueeze(0)
    return x, y, img

def get_accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

imgs, idx = load_data()


optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
optimizer.zero_grad()

def get_lr(i):
    return 2e-5 * (i + 1)

n_epoch = 2
import tqdm
for epoch in range(n_epoch):
    accuracy = 0.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(epoch)

    for i in tqdm.tqdm(range(len(idx))):
        x, y, img = get_data(idx, imgs, i)
        logits = model(x, img)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss = loss / len(idx)
        loss.backward()

        accuracy += get_accuracy(logits, y)

    optimizer.step()
    accuracy /= len(idx)
    print(f"Accuracy: {accuracy}")


accuracy = 0.0
for i in tqdm.tqdm(range(len(idx))):
    x, y, img = get_data(idx, imgs, i)
    logits = model(x, img)
    accuracy += get_accuracy(logits, y)

accuracy /= len(idx)
print(f"Accuracy: {accuracy}")

torch.save(model.state_dict(), "models/hand_math19_finetuned.bin")
