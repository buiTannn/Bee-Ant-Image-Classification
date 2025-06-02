# Bee-Ant Image Classification

Deep learning project for classifying bee and ant images using PyTorch with ResNet18/VGG16 transfer learning.

## Dataset
- **Class 0**: Ant
- **Class 1**: Bee
- Uses Hymenoptera dataset structure

## Files
- `main.py` - Training script
- `test.py` - Inference script  
- `data/` - Dataset folder (train/val splits)
- `models/` - Saved models directory

## Requirements
```bash
pip install torch torchvision pillow
```

## Usage

### Training
```bash
python main.py
```
- Supports ResNet18 and VGG16 backbones
- Auto-saves best model based on accuracy
- 10 epochs, batch size 32, Adam optimizer

### Testing
```bash
python test.py
```
- Loads pre-trained model from `./models/best_model.pth`
- Predicts single image (modify `img_path` variable)
- Returns class: 0 (ant) or 1 (bee)

## Model Architecture
- **Backbone**: ResNet18 (default) or VGG16
- **Classes**: 2 (binary classification)
- **Input**: 224x224 RGB images
- **Output**: Class probabilities


## Data Structure
```
data/
├── train/
│   ├── ants/
│   └── bees/
└── val/
    ├── ants/
    └── bees/
```
