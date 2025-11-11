from torchvision import transforms
import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on {device}")

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


image_path = "/app/dataset/processed/task2/images/050_F_NA1.png"
mask_path  = "/app/dataset/processed/task2/masks/050_M.png"


image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    output    = model(input_tensor)
    pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

pred_bin = (pred_prob > 0.5).astype(np.uint8)




real_mask     = Image.open(mask_path).convert("L").resize(pred_bin.shape[::-1], resample=Image.NEAREST)
real_mask_arr = np.array(real_mask)
real_mask_bin = (real_mask_arr > 127).astype(np.uint8)


y_true = real_mask_bin.flatten()
y_pred = pred_bin.flatten()

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])

print("\n=== Confusion Matrix (Pixels) ===")
print(cm_df)


plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Pixel-Level)")

plt.tight_layout()
plt.savefig("/app/outputs/task2/confusion_matrix.png")
plt.close()

######################################################################

precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)

print("\n=== Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

######################################################################

metrics = {"Precision": precision, "Recall": recall, "F1-Score": f1}

plt.figure(figsize=(5, 4))
plt.bar(metrics.keys(), metrics.values(), color="steelblue")
plt.ylim(0, 1)
plt.title("Model Performance Metric (Pixel-Level)")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("/app/outputs/task2/metrics.png")
plt.close()

######################################################################

y_score = pred_prob.flatten()
roc_auc = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)

print(f"\nROC-AUC: {roc_auc:.4f}")


plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve (Pixel-Level)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/app/outputs/task2/roc_auc.png")
plt.close()