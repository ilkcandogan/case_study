from torchvision import transforms, datasets
from torchvision.models import resnet18
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on {device}")

transform = transforms.Compose([
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


test_dir     = "/app/dataset/processed/task1/test/"
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs >= 0.5).long()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
class_names = test_dataset.classes

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\n=== Confusion Matrix ===")
print(cm_df)


plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("/app/outputs/task1/confusion_matrix.png")
plt.close()

######################################################################

precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print("\n=== Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

######################################################################

metrics = {"Precision": precision, "Recall": recall, "F1-Score": f1}

plt.figure(figsize=(5,4))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.title("Model Performance Metric")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("/app/outputs/task1/metrics.png")
plt.close()

######################################################################

y_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs  = inputs.to(device)
        outputs = model(inputs)
        
        probs = torch.sigmoid(outputs).squeeze()
        y_probs.extend(probs.cpu().numpy())
        
roc_auc = roc_auc_score(y_true, y_probs)
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
print(f"\nROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0,1], [0,1], color="gray", linestyle="--")  
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/app/outputs/task1/roc_auc.png")
plt.close()
