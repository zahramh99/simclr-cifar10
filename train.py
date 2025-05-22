import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from model import SimCLR
from utils import get_simclr_augmentations, nt_xent_loss
from config import config
import os

# Create data directory
os.makedirs(config["data_dir"], exist_ok=True)

# Load CIFAR-10 (no labels)
train_data = datasets.CIFAR10(
    root=config["data_dir"],
    train=True,
    download=True,
    transform=get_simclr_augmentations()
)
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

# Initialize model
model = SimCLR(feature_dim=config["feature_dim"]).to(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Training loop
for epoch in range(config["epochs"]):
    for (x1, x2), _ in train_loader:
        x1, x2 = x1.to(config["device"]), x2.to(config["device"])
        z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2, config["temperature"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "simclr_cifar10.pth")