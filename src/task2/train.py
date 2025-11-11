from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

class ComofodDataset(Dataset):
    def __init__(self, images_dir, mask_dir, transform_image, transform_mask):
        self.images_dir = images_dir
        self.mask_dir   = mask_dir
        
        self.transform_image = transform_image
        self.transform_mask  = transform_mask
        
        self.image_files = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_dir, image_name)
        
        mask_name = image_name.split("_")[0] + "_M.png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        
        mask = (mask > 0).float()
        
        return image, mask
        

transform_image = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

transform_mask = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

train_dataset = ComofodDataset("/app/dataset/processed/task2/images", "/app/dataset/processed/task2/masks", transform_image, transform_mask)

train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])
test_dataset, val_dataset   = random_split(test_dataset, [0.5, 0.5])


train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader   = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader  = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)


model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()


num_epochs = 1 #final model 10
best_val_loss = float('inf')

save_dir = Path("/app/outputs/task2/")
save_dir.mkdir(exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

    for images, masks in train_loader_tqdm:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()
        train_loader_tqdm.set_postfix({"loss": loss.item()})

    train_loss /= len(train_dataloader)


    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

    with torch.no_grad():
        for images, masks in val_loader_tqdm:
            images = images.to(device)
            masks  =  masks.to(device)
            
            outputs = model(images)
            loss    = criterion(outputs, masks)
            
            val_loss += loss.item()
            val_loader_tqdm.set_postfix({"val_loss": loss.item()})

    val_loss /= len(val_dataloader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_dir / "best_unet_model.pth")
        print(f"Best model saved: {save_dir / 'best_unet_model.pth'}")
