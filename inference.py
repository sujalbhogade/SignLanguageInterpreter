import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("asl_model.pth", map_location=device)

classes = checkpoint["classes"]
num_classes = len(classes)

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)

def predict(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(img).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    tensor = tensor.to(device) / 255.0
    tensor = (tensor - MEAN) / STD

    with torch.no_grad():
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1)[0]

    class_id = torch.argmax(probs).item()
    confidence = probs[class_id].item()

    return classes[class_id], confidence
