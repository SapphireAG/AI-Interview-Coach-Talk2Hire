# Import all libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import random
from collections import defaultdict

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("mstjebashazida/affectnet")
class SEBlock(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super(SEBlock, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y.expand_as(x)
            
class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                self.bn2 = nn.BatchNorm2d(out_ch)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                        nn.BatchNorm2d(out_ch)
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return F.relu(out)
class ResEmoteNet(nn.Module):
            def __init__(self, num_classes=7):
                super(ResEmoteNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)

                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)

                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)

                self.se = SEBlock(128, reduction=8)

                self.res_block1 = ResidualBlock(128, 256, stride=2)
                self.res_block2 = ResidualBlock(256, 512, stride=2)
                self.res_block3 = ResidualBlock(512, 1024, stride=2)

                self.pool = nn.AdaptiveAvgPool2d((1, 1))

                self.fc1 = nn.Linear(1024, 512)
                self.bn_fc1 = nn.BatchNorm1d(512)
                self.fc2 = nn.Linear(512, 256)
                self.bn_fc2 = nn.BatchNorm1d(256)
                self.fc3 = nn.Linear(256, 128)
                self.bn_fc3 = nn.BatchNorm1d(128)
                self.fc4 = nn.Linear(128, num_classes)

                self.dropout1 = nn.Dropout(0.3)
                self.dropout2 = nn.Dropout(0.4)

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)

                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)

                x = F.relu(self.bn3(self.conv3(x)))
                x = F.max_pool2d(x, 2)

                x = self.se(x)

                x = self.res_block1(x)
                x = self.res_block2(x)
                x = self.res_block3(x)

                x = self.pool(x)
                x = torch.flatten(x, 1)

                x = F.relu(self.bn_fc1(self.fc1(x)))
                x = self.dropout2(x)
                x = F.relu(self.bn_fc2(self.fc2(x)))
                x = self.dropout2(x)
                x = F.relu(self.bn_fc3(self.fc3(x)))
                x = self.dropout2(x)
                x = self.fc4(x)
                return x
# print("Path to dataset files:", path)
path="/Users/amangolani/.cache/kagglehub/datasets/mstjebashazida/affectnet/versions/1"
emotion_labels = {0: "Happy", 1: "Surprise", 2: "Sad", 3: "Anger", 4: "Disgust", 5: "Fear", 6: "Neutral"}


class AffectNetDataset(Dataset):
    def __init__(self,img_dir,transform=None,oversample_disgust=False,target_disgust_count=6200):
        self.img_dir=img_dir
        self.transform=transform
        self.label_map={'happy':0,'suprise':1,'sad':2,'anger':3,'disgust':4,'fear':5,'neutral':6}
        self.image_data=[]
        class_images=defaultdict(list)
        for emotion in os.listdir(img_dir):
            emotion_path=os.path.join(img_dir,emotion)
            emotion_lower=emotion.lower()
            if os.path.isdir(emotion_path) and emotion_lower in self.label_map:
                label=self.label_map[emotion_lower]
                for filename in os.listdir(emotion_path):
                    if any(filename.endswith(ext) for ext in ['.jpg','.png','.jpeg']):
                        class_images[label].append((os.path.join(emotion_path,filename),label))

        anger_count=len(class_images[3])
        if oversample_disgust and 4 in class_images:
            disgust_data = class_images[4]
            other_data = [(path, label) for label, images in class_images.items() if label != 4 for path, label in images]
            current_disgust_count = len(disgust_data)
            if current_disgust_count > 0:
                oversample_factor = max(1, target_disgust_count // current_disgust_count)
                for _ in range(oversample_factor):
                    self.image_data.extend(disgust_data)
                self.image_data.extend(disgust_data[:target_disgust_count % current_disgust_count])
            self.image_data.extend(other_data)
            for label, images in class_images.items():
                if label != 4:
                    selected_images = random.sample(images, min(anger_count, len(images)))
                    self.image_data.extend(selected_images)
        else:
            for label, images in class_images.items():
                selected_images = random.sample(images, min(anger_count, len(images)))
                self.image_data.extend(selected_images)
        if not self.image_data:
            raise ValueError(f"No images found in {img_dir}")
        
    def __len__(self):
        return len(self.image_data)
    def __getitem__(self,idx):
        img_path,label=self.image_data[idx]
        image=Image.open(img_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image,label

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
if __name__ == "__main__":
        base_path_affectnet = path
        train_img_dir_affectnet = os.path.join(base_path_affectnet,"archive3" , "Train")
        test_img_dir_affectnet = os.path.join(base_path_affectnet,"archive3", "Test")



        affectnet_train_dataset = AffectNetDataset(img_dir=train_img_dir_affectnet, transform=train_transforms, oversample_disgust=True, target_disgust_count=4000)

        train_split=int(0.8*len(affectnet_train_dataset))
        val_split=len(affectnet_train_dataset)-train_split
        train_dataset,val_dataset=random_split(affectnet_train_dataset,[train_split,val_split],
                                                generator=torch.Generator().manual_seed(42))

        affectnet_test_dataset=AffectNetDataset(img_dir=test_img_dir_affectnet,transform=test_transforms)
        print("Length of the FULL training dataset",affectnet_train_dataset)
        print("Validation Split",len(val_dataset))
        print("Train Split",len(train_dataset))

        train_original_labels=[]
        train_original_labels.extend(label for _, label in affectnet_train_dataset.image_data)

        train_class_counts = pd.Series(train_original_labels).value_counts().sort_index()
        print("\nTraining set class distribution (Mapped to target emotion labels 0-6):")
        for mapped_label, count in train_class_counts.items():
                emotion_name = emotion_labels[mapped_label]
                print(f"Class {mapped_label} ({emotion_name}): {count} images")

            # DataLoaders with increased batch size
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(affectnet_test_dataset, batch_size=64, shuffle=False, num_workers=4)

            # Batch shape check
        train_image, train_label = next(iter(train_loader))
        print(f"\nTrain batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
        val_image, val_label = next(iter(val_loader))
        print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
        test_image, test_label = next(iter(test_loader))
        print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

        class SEBlock(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super(SEBlock, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y.expand_as(x)
            
        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                self.bn2 = nn.BatchNorm2d(out_ch)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                        nn.BatchNorm2d(out_ch)
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return F.relu(out)
        class ResEmoteNet(nn.Module):
            def __init__(self, num_classes=7):
                super(ResEmoteNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)

                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(128)

                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(256)

                self.se = SEBlock(256)

                self.res_block1 = ResidualBlock(256, 512, stride=2)
                self.res_block2 = ResidualBlock(512, 1024, stride=2)
                self.res_block3 = ResidualBlock(1024, 2048, stride=2)

                self.pool = nn.AdaptiveAvgPool2d((1, 1))

                self.fc1 = nn.Linear(2048, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, 256)
                self.fc4 = nn.Linear(256, num_classes)

                self.dropout1 = nn.Dropout(0.3)
                self.dropout2 = nn.Dropout(0.4)

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)

                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)

                x = F.relu(self.bn3(self.conv3(x)))
                x = F.max_pool2d(x, 2)

                x = self.se(x)

                x = self.res_block1(x)
                x = self.res_block2(x)
                x = self.res_block3(x)

                x = self.pool(x)
                x = torch.flatten(x, 1)

                x = F.relu(self.fc1(x))
                x = self.dropout2(x)
                x = F.relu(self.fc2(x))
                x = self.dropout2(x)
                x = F.relu(self.fc3(x))
                x = self.dropout2(x)
                x = self.fc4(x)
                return x
            

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResEmoteNet(num_classes=7).to(device)

        # Step 5: Print total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")

        # Class weights
        class_counts = np.array([8540, 7495, 9291, 7700, 7341, 7288, 8958])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # Step 6: Define loss with label smoothing for training
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')

        def mixup_data(x, y, alpha=0.2):
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        # Step 7: Checkpoint Management
        checkpoint_dir = "/Users/amangolani/FER"
        os.makedirs(checkpoint_dir, exist_ok=True)

        start_epoch = 0
        best_acc = 0.0
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_accs = checkpoint.get('val_accs', [])
            print(f"Resumed training from epoch {start_epoch} with best accuracy {best_acc:.2f}%")


        patience = 30
        epochs_no_improve = 0
        num_epochs = 200
        best_model_path = "/Users/amangolani/FER/best_resemotenet_model.pth"
        last_checkpoint_path = checkpoint_path

        for epoch in range(start_epoch, num_epochs):
            # Training Phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
            for imgs, labels in train_progress:
                imgs, labels = imgs.to(device), labels.to(device)
                # Apply MixUp
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=1.0)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}. Skipping batch.")
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels_a).sum().item()  # Use labels_a for accuracy
                train_progress.set_postfix({'loss': running_loss / (len(train_progress) + 1)})

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation Phase (use regular CrossEntropyLoss for validation)
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            with torch.no_grad():
                for images, labels in val_progress:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)  # Regular CE for validation
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_progress.set_postfix({'loss': val_running_loss / (len(val_progress) + 1)})

            val_loss = val_running_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Print summary
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
            torch.save(checkpoint, last_checkpoint_path)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Val Acc: {best_acc:.2f}%")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

            scheduler.step()