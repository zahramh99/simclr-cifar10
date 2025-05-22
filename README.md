# SimCLR on CIFAR-10 :sunrise_over_mountains:

Train a self-supervised vision model (SimCLR) on CIFAR-10 **without labels**, then evaluate learned features on image classification.

## **How to Run**
```bash
# Install dependencies
pip install -r requirements.txt

# Train SimCLR (self-supervised pretraining)
python train.py

# Evaluate on downstream task (linear probing)
python evaluate.py

Results
After training, check:

results/ for loss curves and feature visualizations.

Test accuracy (~75-85% with linear probing on CIFAR-10).

Key Features
:zap: Pure PyTorch implementation

:mag_right: Supports custom datasets (edit config.py)

:chart_with_upwards_trend: Logs metrics to TensorBoard (optional)