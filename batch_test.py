import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn

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
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.model(x)

model = CNN()
model.load_state_dict(torch.load("traffic_light_model.pth"))
model.eval()

classes = ['back', 'green', 'red', 'yellow']

transform = transforms.Compose([
    transforms.Resize((180,180)),
    transforms.ToTensor()
])

folder = "test_images"

for img_name in os.listdir(folder):
    img_path = os.path.join(folder, img_name)

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    output = model(img)
    pred = torch.argmax(output)

    print(img_name, "→", classes[pred])