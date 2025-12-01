# HRI30 Video Action Recognition - PyTorch Implementation

Complete PyTorch implementation for action recognition on the HRI30 dataset using SlowOnly architecture with pre-training and fine-tuning support.

## 📋 Overview

This project implements state-of-the-art video action recognition for the HRI30 industrial human-robot interaction dataset:

- **30 action classes** of industrial tasks
- **SlowOnly** architecture (achieves 86.55% accuracy per paper)
- **Pre-trained backbone** (ImageNet/Kinetics-400)
- **Complete pipeline**: data loading → training → validation → testing → fine-tuning
- **Production-ready**: checkpointing, early stopping, learning rate scheduling

## 🎯 Key Features

✅ **SlowOnly Model**: State-of-the-art for video action recognition
✅ **Transfer Learning**: Pre-trained ImageNet backbone
✅ **Data Augmentation**: Random flip, brightness/contrast adjustment
✅ **Comprehensive Training**: Validation split, metrics tracking
✅ **Early Stopping**: Prevent overfitting
✅ **Learning Rate Scheduling**: Cosine annealing, step decay, exponential decay
✅ **Fine-tuning Strategies**: 3 different approaches for model adaptation
✅ **Detailed Evaluation**: Per-class metrics, confusion matrix, top-5 accuracy
✅ **Visualization**: Training curves, confusion matrix plots
✅ **Test Predictions**: CSV export with confidence scores and top-5

## 📦 Project Structure

```
project/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Video loading and augmentation
├── model.py               # SlowOnly and CNN-LSTM architectures
├── train.py               # Main training script
├── evaluate.py            # Evaluation and prediction
├── fine_tune.py           # Fine-tuning strategies
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
│
├── checkpoints/           # Saved model checkpoints
├── results/               # Evaluation results and plots
├── logs/                  # Training logs
│
└── README.md             # This file
```

## ⚙️ Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Fix CUDA Issue (IMPORTANT FOR YOUR SYSTEM)

Your system reports `CUDA available: False` even with GPU installed. This is likely an environment issue.

**Solution:**

```bash
# Verify GPU availability first
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify again
python3 -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Verify Dataset Structure

Ensure your data is organized as:

```
/home/marino/groupSAP/data/
├── train/
│   ├── CID01_SID01_VID01.avi
│   ├── CID01_SID01_VID02.avi
│   ├── CID02_SID02_VID01.avi
│   └── ...
└── test/
    ├── video_001.avi
    ├── video_002.avi
    └── ...
```

The training videos **must** follow the naming format `CIDXX_SIDYY_VIDZZ.avi` to extract class labels.

## 🚀 Quick Start

### Step 1: Training

```bash
python train.py
```

This will:
- Load ~1800 training videos
- Split into 90% train / 10% validation
- Train SlowOnly model for up to 100 epochs
- Save best model to `checkpoints/hri30_slowonly_best.pt`
- Plot training curves

**Expected runtime**: ~6-8 hours on RTX 5070 Ti (adjust BATCH_SIZE if needed)

**Config options** (in `config.py`):
- `BATCH_SIZE`: 16 (reduce to 8 if out of memory)
- `NUM_EPOCHS`: 100
- `INITIAL_LR`: 0.001
- `LR_SCHEDULER`: "cosine" (options: cosine, step, exponential)

### Step 2: Evaluation

```bash
python evaluate.py
```

This will:
- Evaluate on validation set
- Make predictions on ~700 test videos
- Save predictions to `results/test_predictions.csv`
- Generate confusion matrix
- Print per-class metrics

### Step 3: Fine-tuning (Optional)

After initial training, you can fine-tune:

```bash
python fine_tune.py
```

**Three strategies available:**

1. **Strategy 1**: Freeze backbone, train only head (fastest, good for small data)
2. **Strategy 2**: Differential learning rates (balanced, recommended)
3. **Strategy 3**: Progressive unfreezing (slowest, best results)

Edit line in `fine_tune.py` to select strategy:
```python
strategy = 2  # Change to 1, 2, or 3
```

## 📊 Understanding Key Concepts

### Learning Rate Scheduling

Learning rate schedulers adjust the learning rate during training:

- **Cosine Annealing** (recommended): Gradually decreases LR using cosine curve, helps escape local minima
- **Step Decay**: Drops LR by factor after N epochs
- **Exponential Decay**: Multiplies LR by factor each epoch

**Why?** Starting with high LR helps explore, lowering LR helps fine-tune.

### Fine-tuning vs Training from Scratch

**Pre-trained backbone** means model already learned basic image features:
- ✅ Much faster training (5-10x speedier)
- ✅ Better accuracy with limited data
- ✅ Requires less computational power

**Fine-tuning strategies:**
1. **Freeze backbone** = only adjust final layers (safest)
2. **Low LR for backbone** = gently adapt pre-trained features (balanced)
3. **Progressive unfreezing** = gradually unfreeze layers (best)

### Early Stopping

Prevents overfitting by stopping training when validation loss stops improving:
```python
EARLY_STOPPING_PATIENCE = 15      # Stop after 15 epochs with no improvement
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to count as improvement
```

## 📈 Monitoring Training

### Training Curves

After training, `results/training_curves.png` shows:
- **Left**: Train vs Validation Loss
- **Right**: Train vs Validation Accuracy

Good training = loss decreasing, curves close together

### Confusion Matrix

`results/confusion_matrix.png` shows which classes are confused:
- Dark diagonal = good (correctly classified)
- Dark off-diagonal = classes being confused

### CSV Predictions

`results/test_predictions.csv` format:
```
Video,Predicted_Class,Predicted_Label,Confidence,Top_5
video_001.avi,25,UsingTheDrill,0.9823,"UsingTheDrill(0.9823); UsingThePolisher(0.0142); ..."
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

```python
# In config.py, reduce batch size
BATCH_SIZE = 8  # Was 16
```

### Training too slow

```python
# Skip validation plotting
PLOT_LOSS = False

# Reduce data augmentation (in data_loader.py)
RANDOM_FLIP = False
```

### Model not improving

1. Check learning rate: try `INITIAL_LR = 0.01` or `0.0001`
2. Use different scheduler: try `LR_SCHEDULER = "step"`
3. Increase training time: raise `NUM_EPOCHS`
4. Use fine-tuning strategy for better pre-training adaptation

### CUDA still not working

```bash
# Force CPU training (slower but works)
# In train.py, change:
USE_GPU = False

python train.py
```

## 📝 Configuration Guide

Edit `config.py` to customize:

```python
# Dataset
NUM_CLASSES = 30
TARGET_FRAMES = 8  # Frames per video

# Model
BACKBONE = "resnet50"           # Architecture
PRETRAINED_DATASET = "kinetics400"  # Pre-training source

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 100
INITIAL_LR = 0.001
LR_SCHEDULER = "cosine"

# Optimization
OPTIMIZER = "sgd"           # sgd or adam
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Regularization
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

# Resources
USE_GPU = True
GPU_ID = 0
NUM_WORKERS = 4  # Data loading threads
```

## 📚 Understanding the Code

### data_loader.py

- **HRI30VideoDataset**: Loads videos, samples frames uniformly, applies augmentation
- **Frame sampling**: Uniformly samples N frames from variable-length videos
- **Augmentation**: Random flip, brightness adjustment during training

### model.py

- **SlowOnlyModel**: ResNet-50 backbone + temporal pooling + FC head
- **CNNLSTMModel**: Alternative CNN-LSTM architecture (slower, more memory)

### train.py

- **Trainer**: Handles training loop, checkpoint saving, early stopping
- **Optimizer setup**: SGD with momentum or Adam
- **Scheduler**: Cosine annealing / step / exponential

### evaluate.py

- **Evaluator**: Makes predictions, computes metrics
- **Top-K predictions**: Shows 5 most likely classes
- **CSV export**: Prediction results with confidence

### fine_tune.py

- **3 fine-tuning strategies**: Freeze backbone, differential LR, progressive unfreezing
- **Transfer learning**: Adapts pre-trained model to HRI30

## 🎓 Expected Performance

Based on HRI30 paper:

- **SlowOnly (Kinetics-400)**: ~86.55% accuracy on 30 classes
- **Your system**: ~70-80% expected initially (due to dataset variations)
- **After fine-tuning**: Can improve 5-10%

Per-class accuracy varies:
- Easy classes (Walking, UsingTheDrill): 90%+
- Hard classes (similar movements): 60-70%

## 💡 Tips for Better Results

1. **More training data**: Larger dataset = better results
2. **Data augmentation**: Enable more augmentations in data_loader.py
3. **Longer training**: Increase NUM_EPOCHS
4. **Lower learning rate**: INITIAL_LR = 0.0005 helps with smaller datasets
5. **Fine-tuning**: Use Strategy 3 (progressive unfreezing)
6. **Ensemble**: Train multiple models, average predictions

## 📖 References

- Paper: "SlowFast Networks for Video Recognition" (ICCV 2019)
- HRI30 Dataset: https://zenodo.org/record/5833411
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

## 📞 Support

Common issues and solutions in the **Troubleshooting** section above.

For specific problems:
1. Check config.py matches your setup
2. Verify dataset paths and file structure
3. Check GPU availability with: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

**Last Updated**: November 2025
**PyTorch Version**: 2.9.1+cu130
**Python**: 3.11
**Status**: Production Ready ✅
# GropSap
# GropSap
