import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from model import SimCLR
from config import config

# Load pretrained SimCLR encoder
model = SimCLR(feature_dim=config["feature_dim"])
model.load_state_dict(torch.load("simclr_cifar10.pth"))
encoder = model.encoder  # Freeze this and train a classifier on top

# Add linear layer
classifier = nn.Linear(2048, 10).to(config["device"])  # CIFAR-10 has 10 classes

# Train classifier (example snippet)
train_data = datasets.CIFAR10(
    root=config["data_dir"],
    train=True,
    download=True,
    transform=transforms.ToTensor()  # No augmentations needed
)
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

for epoch in range(10):
    for images, labels in train_loader:
        features = encoder(images.to(config["device"]))  # Extract features
        outputs = classifier(features)
        loss = nn.CrossEntropyLoss()(outputs, labels.to(config["device"]))
        # Backpropagate (only updates classifier, not encoder!)
        ...