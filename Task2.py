# Task 2: Image Calssification using Pytorch

# Importing required libraries
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset for unlabeled images
class UnlabeledImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [os.path.join(folder_path, file)
                            for file in os.listdir(folder_path)
                            if file.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]

# Extract features using pretrained SqueezeNet
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(squeezenet.features.children()))

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            return features.view(features.size(0), -1)

# Cluster features and assign pseudo-labels
def cluster_and_label(dataset, feature_extractor, n_clusters=5):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_features, all_paths = [], []
    for imgs, paths in tqdm(dataloader, desc="Extracting features"):
        feats = feature_extractor(imgs)
        all_features.append(feats)
        all_paths.extend(paths)
    all_features = torch.cat(all_features).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(all_features)
    return list(zip(all_paths, labels)), kmeans, n_clusters

# Dataset with pseudo-labels
def create_labeled_dataset(pseudo_labeled_data, transform):
    class LabeledDataset(Dataset):
        def __init__(self, labeled_data):
            self.data = labeled_data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = Image.open(self.data[idx][0]).convert("RGB")
            label = self.data[idx][1]
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)  # FIXED dtype

    return LabeledDataset(pseudo_labeled_data)

# Train classifier (SqueezeNet)
def train_classifier(model, train_loader, val_loader, num_classes, device, num_epochs=5):
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader.dataset):.4f} | Acc: {acc:.4f}")

    return model

# Predict single image
def predict_image(image_path, model, transform, kmeans, feature_extractor, device):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(img_tensor).cpu().numpy()
        cluster_label = kmeans.predict(feature)[0]
    print(f"Predicted Cluster Label: {cluster_label}")
    plt.imshow(image)
    plt.title(f"Predicted Cluster: {cluster_label}")
    plt.axis("off")
    plt.savefig(f"predicted_cluster_{cluster_label}.png")
    plt.show()

# Main
if __name__ == "__main__":
    set_seed()
    folder = input("Enter path to training image folder: ").strip()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    raw_dataset = UnlabeledImageDataset(folder, transform)
    feature_extractor = FeatureExtractor()
    pseudo_labeled_data, kmeans_model, n_clusters = cluster_and_label(raw_dataset, feature_extractor)
    labeled_dataset = create_labeled_dataset(pseudo_labeled_data, transform)
    val_size = int(0.2 * len(labeled_dataset))
    train_size = len(labeled_dataset) - val_size
    train_data, val_data = random_split(labeled_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    trained_model = train_classifier(classifier, train_loader, val_loader, num_classes=n_clusters, device=device)

    test_image = input("Enter test image path: ").strip()
    predict_image(test_image, trained_model, transform, kmeans_model, feature_extractor, device)
