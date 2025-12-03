# Utility functions for HRI30 project
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import json
from config import CLASS_NAMES, CLASS_TO_IDX, RESULTS_DIR, LOGS_DIR


def parse_hri30_filename(filename):
    """
    Parse HRI30 filename format: CIDXX_SIDYY_VIDZZ.avi
    Returns: (class_id, subject_id, video_id)
    
    Example: CID01_SID03_VID05.avi -> (1, 3, 5)
    """
    # Remove extension
    name = Path(filename).stem
    
    # Match pattern: CID##_SID##_VID##
    pattern = r"CID(\d+)_SID(\d+)_VID(\d+)"
    match = re.match(pattern, name)
    
    if match:
        class_id = int(match.group(1))
        subject_id = int(match.group(2))
        video_id = int(match.group(3))
        return class_id, subject_id, video_id
    
    return None, None, None


def get_class_id_from_filename(filename):
    """Extract class ID from filename"""
    class_id, _, _ = parse_hri30_filename(filename)
    return class_id


def get_all_video_files(directory, extension=".avi"):
    """Get all video files from directory"""
    directory = Path(directory)
    if not directory.exists():
        print(f"⚠ Directory not found: {directory}")
        return []

    ext = extension.lower()
    return sorted([
        f.name for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() == ext
    ])


def plot_training_history(train_losses, val_losses, train_accs=None, val_accs=None, 
                          save_path=None, title="Training History"):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(14, 5))
    
    if train_accs is None:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(train_losses, label="Train Loss", marker="o", markersize=4)
    axes[0].plot(val_losses, label="Validation Loss", marker="s", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot (if provided)
    if train_accs is not None:
        axes[1].plot(train_accs, label="Train Accuracy", marker="o", markersize=4)
        axes[1].plot(val_accs, label="Validation Accuracy", marker="s", markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy over Epochs")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([class_names[i+1][:10] for i in range(len(class_names))], rotation=90, fontsize=8)
    ax.set_yticklabels([class_names[i+1][:10] for i in range(len(class_names))], fontsize=8)
    
    # Normalize
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot text
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                   color="white" if cm[i, j] > threshold else "black", fontsize=6)
    
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Confusion matrix saved: {save_path}")
    
    plt.show()


def save_predictions_to_csv(results, save_path):
    """
    Save predictions to CSV file
    
    results: list of dicts with keys:
        - video_name
        - predicted_class
        - predicted_label
        - confidence
        - top_5_predictions (list of tuples: (class_idx, class_name, confidence))
    """
    import pandas as pd
    
    data = []
    for result in results:
        top5_classes = ";".join([name for _, name, _ in result["top_5"]])
        top5_probs = ";".join([f"{conf:.4f}" for _, _, conf in result["top_5"]])
        data.append({
            "video_name": result["video_name"],
            "top1_idx": result["predicted_class"],
            "top1_class": result["predicted_label"],
            "top1_prob": f"{result['confidence']:.4f}",
            "top5_classes": top5_classes,
            "top5_probs": top5_probs,
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✓ Predictions saved: {save_path}")
    return df


def save_metrics_to_json(metrics, save_path):
    """Save metrics to JSON"""
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {save_path}")


def setup_device(use_gpu=True, gpu_id=0):
    """Setup device (CPU or GPU)"""
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    return device


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"✓ Random seed set to {seed}")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, mode="min"):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as an improvement
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = -float("inf")
    
    def __call__(self, current_value):
        """
        Check if training should stop
        Returns: True if should stop, False otherwise
        """
        if self.mode == "min":
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
                return False
        else:  # max
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
                return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class AverageMeter:
    """Track average of a metric"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics(metrics, stage="Validation"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{stage} Metrics")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {key:.<40} {value}")
    print(f"{'='*60}\n")
