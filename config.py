# Configuration for HRI30 Video Action Recognition

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path("/home/marino/groupSAP/data")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET
# ============================================================================
NUM_CLASSES = 30
CLASS_NAMES = {
    1: "DeliverObject",
    2: "MoveBackwardsWhileDrilling",
    3: "MoveBackwardsWhilePolishing",
    4: "MoveDiagonallyBackwardLeftWithDrill",
    5: "MoveDiagonallyBackwardLeftWithPolisher",
    6: "MoveDiagonallyBackwardRightWithDrill",
    7: "MoveDiagonallyBackwardRightWithPolisher",
    8: "MoveDiagonallyForwardLeftWithDrill",
    9: "MoveDiagonallyForwardLeftWithPolisher",
    10: "MoveDiagonallyForwardRightWithDrill",
    11: "MoveDiagonallyForwardRightWithPolisher",
    12: "MoveForwardWhileDrilling",
    13: "MoveForwardWhilePolishing",
    14: "MoveLeftWhileDrilling",
    15: "MoveLeftWhilePolishing",
    16: "MoveRightWhileDrilling",
    17: "MoveRightWhilePolishing",
    18: "NoCollaborativeWithDrill",
    19: "NoCollaborativeWithPolisher",
    20: "PickUpDrill",
    21: "PickUpPolisher",
    22: "PickUpTheObject",
    23: "PutDownDrill",
    24: "PutDownPolisher",
    25: "UsingTheDrill",
    26: "UsingThePolisher",
    27: "Walking",
    28: "WalkingWithObject",
    29: "WalkingWithDrill",
    30: "WalkingWithPolisher",
}

# Reverse mapping
CLASS_TO_IDX = {v: k for k, v in CLASS_NAMES.items()}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
VIDEO_EXTENSION = ".avi"
TARGET_FRAMES = 24  # Number of frames to sample per video
FRAME_RATE = 30  # HRI30 has 30 FPS
INPUT_SIZE = (224, 224)  # Input resolution for SlowOnly
CLIP_DURATION = 1  # Duration in seconds to consider for fast path (SlowOnly specific)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
BACKBONE = "resnet50"  # Backbone architecture
MODEL_ARCH = "slowonly"  # Main architecture (slowonly, slowfast, cnn_lstm)
PRETRAINED_DATASET = "r2plus1d_18"  # Options: r2plus1d_18, r3d_18, mc3_18, imagenet, None

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 4  # Reduced for 24 frames + LSTM (was 16 for 8 frames)
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate batch_size=16
NUM_WORKERS = 4  # DataLoader workers
NUM_EPOCHS = 100  # Maximum epochs
INITIAL_LR = 0.0003  # For Adam optimizer (Adam needs lower LR than SGD)
LR_SCHEDULER = "cosine"  # Learning rate scheduler: cosine, step, exponential
LR_WARMUP_EPOCHS = 5  # Warmup epochs before full LR
LR_STEP_SIZE = 30  # Step size for step scheduler
LR_GAMMA = 0.1  # Decay rate for step/exponential scheduler

# Optimizer
OPTIMIZER = "adam"  # sgd or adam
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4  # Reduced for Adam (Adam has built-in regularization)

# ============================================================================
# TRAINING STRATEGIES
# ============================================================================
EARLY_STOPPING_PATIENCE = 10  # Increased patience to allow more training
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to count as improvement

# Data augmentation
RANDOM_FLIP = False  # Keep False for action recognition (flipping changes action direction)
RANDOM_CROP = False  # Enable random crop for better generalization
CROP_SCALE = (0.8, 1.0)  # Random crop scale range
COLOR_JITTER = True  # Enable color jitter
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
NORMALIZE_MEAN = [0.45, 0.45, 0.45]  # ImageNet mean
NORMALIZE_STD = [0.225, 0.225, 0.225]  # ImageNet std

# Label smoothing for regularization
LABEL_SMOOTHING = 0.0  # Disabled temporarily to allow faster learning

# Memory optimization
USE_MIXED_PRECISION = True  # Use FP16 to reduce memory usage (2x less memory)
CLEAR_CACHE_EVERY_N_BATCHES = 10  # Clear GPU cache periodically

# ============================================================================
# VALIDATION & TESTING
# ============================================================================
VALIDATION_SPLIT = 0.1  # 10% of training data for validation
TOP_K_PREDICTIONS = 5  # Show top-5 predictions

# ============================================================================
# DEVICE
# ============================================================================
USE_GPU = True
GPU_ID = 0  # GPU device index

# ============================================================================
# CHECKPOINTING
# ============================================================================
SAVE_CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
KEEP_BEST_CHECKPOINT = True  # Keep only the best checkpoint
CHECKPOINT_PREFIX = "hri30_slowonly"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False  # Set to True for faster training if deterministic not needed

# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================
LOG_INTERVAL = 10  # Print stats every N batches
PLOT_LOSS = True  # Plot training/validation loss curves
SAVE_PLOTS = True  # Save plots to disk
CONFUSION_MATRIX = True  # Generate confusion matrix on test set

print("=" * 70)
print("HRI30 Configuration Loaded")
print(f"Dataset: Train={TRAIN_DIR} | Test={TEST_DIR}")
print(f"Classes: {NUM_CLASSES}")
print(f"Model: {MODEL_ARCH.upper()} with {BACKBONE.upper()} backbone")
print(f"Pre-trained: {PRETRAINED_DATASET.upper()}")
print(f"Batch Size: {BATCH_SIZE} | LR: {INITIAL_LR} | Epochs: {NUM_EPOCHS}")
print("=" * 70)
