import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("traffic_light_data/train", transform=transform)
val_data = datasets.ImageFolder("traffic_light_data/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

classes = train_data.classes
print("Classes:", classes)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, len(classes))
        )

    def forward(self, x):
        return self.model(x)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "traffic_light_model.pth")
print("Model saved!")