import os
import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

# ===== CONFIG =====
TEST_DIR = "asl_alphabet_test/asl_alphabet_test"
MODEL_PATH = "asl_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD MODEL =====
checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint["classes"]

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# ===== NORMALIZATION =====
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# ===== PREDICT =====
def predict(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device) / 255.0
    tensor = (tensor - MEAN) / STD

    with torch.no_grad():
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1)[0]

    class_id = torch.argmax(probs).item()
    confidence = probs[class_id].item()

    return classes[class_id], confidence

# ===== TEST LOOP =====
correct = 0
total = 0

print("\n===== TEST RESULTS =====\n")

for file in sorted(os.listdir(TEST_DIR)):
    if not file.endswith(".jpg"):
        continue

    # Extract label
    true_label = file.split("_")[0]

    img_path = os.path.join(TEST_DIR, file)
    img = cv2.imread(img_path)

    pred_label, conf = predict(img)

    is_correct = pred_label == true_label

    print(f"{file:20} | Pred: {pred_label:8} | True: {true_label:8} | Conf: {conf:.2f} | {'✓' if is_correct else '✗'}")

    correct += int(is_correct)
    total += 1

# ===== FINAL ACCURACY =====
accuracy = correct / total
print("\n========================")
print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
