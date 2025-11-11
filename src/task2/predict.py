from torchvision import transforms
import segmentation_models_pytorch as smp
import torch
import sys
import random
from PIL import Image
import numpy as np
import shutil

image_path = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Prediction on {device}")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)
model.eval()


checkpoint = torch.load("/app/models/final_task2_unet_model.pth", map_location=device)
model.load_state_dict(checkpoint)

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    output    = model(input_tensor)
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255


Image.fromarray(pred_mask).save("/app/outputs/task2/pred_mask.png")
print("Prediction mask saved: /app/outputs/task2/pred_mask.png")

shutil.copy(image_path, "/app/outputs/task2/pred_image.png")