import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path

# ================= CONFIG =================
DATA_DIR = "asl_alphabet_train/asl_alphabet_train"
BATCH_SIZE = 16
VAL_SPLIT = 0.2
EPOCHS = 10
SEED = 42

accumulation_steps = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= DATASET =================
full_dataset = datasets.ImageFolder(DATA_DIR)
num_classes = len(full_dataset.classes)

total_size = len(full_dataset)
val_size = int(total_size * VAL_SPLIT)
train_size = total_size - val_size

generator = torch.Generator().manual_seed(SEED)

train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

# ===== wrapper to apply transforms properly =====
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)

train_dataset = TransformSubset(train_subset, train_transform)
val_dataset = TransformSubset(val_subset, val_transform)

# ================= LOADERS =================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ================= MODEL =================
model = efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ================= TRAINING =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        running_loss += loss.item() * accumulation_steps  # restore true loss

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    #IMPORTANT: handle leftover gradients
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    val_acc = evaluate()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f}")

# ================= SAVE =================
torch.save({
    "model_state": model.state_dict(),
    "classes": full_dataset.classes
}, "asl_model.pth")

print("Training complete. Model saved as asl_model.pth")
