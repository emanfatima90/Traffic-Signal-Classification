import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= DATA =================
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor()
])

val_data = datasets.ImageFolder(
    "traffic_light_data/val",
    transform=transform
)

val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False
)

classes = val_data.classes

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*20*20,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        return self.model(x)

# ================= LOAD MODEL =================
model = CNN()
model.load_state_dict(
    torch.load("traffic_light_model.pth", map_location="cpu")
)
model.eval()

# ================= EVALUATION =================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# ================= RESULTS =================
accuracy = accuracy_score(all_labels, all_preds)

print("\n==============================")
print(f"Validation Accuracy: {accuracy*100:.2f}%")
print("==============================\n")

print("Classification Report:\n")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)