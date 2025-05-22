import torch

config = {
    "data_dir": "./data",          # Where CIFAR-10 is downloaded
    "batch_size": 256,             # Reduce if GPU runs out of memory
    "epochs": 100,                 # SimCLR needs longer training
    "lr": 3e-4,                    # Learning rate
    "feature_dim": 128,            # Output feature size
    "temperature": 0.5,            # For NT-Xent loss
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
