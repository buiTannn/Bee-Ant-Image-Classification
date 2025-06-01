# PyTorch Image Classifier - Ant vs Bee
# Refactor code from PyTorch homepage
Image classification project using PyTorch with ResNet18 and VGG16 backbones to classify ants and bees.

## Requirements

```bash
pip install torch torchvision pillow
```

## Dataset Structure

```
data/
├── train/
│   ├── ants/
│   └── bees/
└── val/
    ├── ants/
    └── bees/
```

## Usage

### Training (main.py)
```bash
python main.py
```

- Supports ResNet18 and VGG16 backbones
- Uses data augmentation and saves best model
- Default: resnet18, 10 epochs, batch size 32

### Testing (test.py)
```bash
python test.py
```

- Loads saved model for inference
- Predicts single image class
- Default: ResNet18 model

## Key Features

- **Model Class**: Easy backbone switching between ResNet18/VGG16
- **Learner Class**: Handles training, testing, and inference
- **Auto-save**: Saves best performing model during training
- **GPU Support**: Automatically uses CUDA if available

## Output

- Training: Saves model to `./models/best_model.pth`
- Classes: 0 = ant, 1 = bee
- Console output shows training progress and test accuracy
