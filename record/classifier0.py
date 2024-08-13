"""
Trained on images with only one character.
After 1000 iteration with batch of 512:
epoch: 990, loss: 1.1812726259231567, accuracy: 0.689453125
epoch: 991, loss: 0.8996517658233643, accuracy: 0.7890625
epoch: 992, loss: 0.6970160007476807, accuracy: 0.837890625
epoch: 993, loss: 1.238792896270752, accuracy: 0.6953125
epoch: 994, loss: 0.6015772223472595, accuracy: 0.875
epoch: 995, loss: 1.046127438545227, accuracy: 0.728515625
epoch: 996, loss: 0.642776608467102, accuracy: 0.869140625
epoch: 997, loss: 0.6371352672576904, accuracy: 0.869140625
epoch: 998, loss: 0.8454564809799194, accuracy: 0.791015625
epoch: 999, loss: 0.5196998715400696, accuracy: 0.91796875
"""

class SimpleCNN(nn.Module):
    def __init__(self, n_embed):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 32, 512)  # Adjusted based on the final output size after convolutions
        self.fc2 = nn.Linear(512, n_embed)
        
    def forward(self, x):
        # Convolution -> Activation -> Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 64, 128)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (32, 32, 64)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (64, 16, 32)
        
        # Flatten the tensor
        x = x.view(-1, 64 * 16 * 32)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

model = SimpleCNN(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for i in range(1000):
    optimizer.zero_grad()
    x, img, y = load_data(images, idx_data, 512, device)
    logits = model(img)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    accuracy = cal_accuracy(logits, y)
    print(f"epoch: {i}, loss: {loss.item()}, accuracy: {accuracy}")

torch.save(model.state_dict(), "classifier.bin")