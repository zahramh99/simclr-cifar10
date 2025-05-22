import torch
import torch.nn as nn
from torchvision import transforms

def get_simclr_augmentations():
    return transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.ToTensor(),
    ])

def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    pos = torch.cat([torch.diag(sim, n//2), torch.diag(sim, -n//2)]).view(n, 1)
    neg = sim[mask].reshape(n, -1)
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(n, dtype=torch.long, device=z.device)
    return nn.CrossEntropyLoss()(logits, labels)