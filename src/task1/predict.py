from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch
from PIL import Image
import sys

image_path = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Predicting on {device}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc     = nn.Linear(num_features, 1)

model = model.to(device)
model.eval()

model.load_state_dict(torch.load("/app/models/final_task1_resnet18_model.pth", map_location=device))


image        = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)
input_tensor = input_tensor.to(device)


with torch.no_grad():
    outputs = model(input_tensor)
    prob    = torch.sigmoid(outputs).item()


if prob >= 0.5:
    predicted_label = "real"
else:
    predicted_label = "fake"

print(f"Predicted: {predicted_label} (confidence: {prob:.4f})")
