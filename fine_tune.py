# Fine-tuning script for HRI30

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pathlib import Path

from config import (
    CHECKPOINT_DIR, CHECKPOINT_PREFIX, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS,
    VALIDATION_SPLIT, USE_GPU, GPU_ID, EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA
)

from data_loader import create_dataloaders
from model import SlowOnlyModel
from utils import setup_device, set_seed, EarlyStopping, AverageMeter


class FineTuner:
    """Fine-tune a pre-trained model"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
    
    def freeze_backbone(self, freeze_until_layer=None):
        """
        Freeze backbone layers for fine-tuning
        
        Args:
            freeze_until_layer: Freeze all layers except the last N layers
                               If None, freeze all backbone, only train head
        """
        # Freeze all feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        print("✓ Backbone frozen (requires_grad=False)")
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers"""
        for param in self.model.features.parameters():
            param.requires_grad = True
        
        print("✓ Backbone unfrozen (requires_grad=True)")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def setup_optimizer(self, learning_rate=0.0001, weight_decay=1e-4):
        """
        Setup optimizer with different learning rates for different layers
        
        This is a common fine-tuning strategy:
        - Lower learning rate for backbone (already pre-trained)
        - Higher learning rate for head (being trained from scratch/scratch)
        """
        
        # Separate backbone and head parameters
        backbone_params = list(self.model.features.parameters())
        head_params = [p for p in self.model.parameters() if p not in backbone_params]
        
        # Create optimizer with different learning rates
        optimizer = optim.SGD([
            {'params': backbone_params, 'lr': learning_rate},  # Lower LR for backbone
            {'params': head_params, 'lr': learning_rate * 10}  # Higher LR for head
        ], momentum=0.9, weight_decay=weight_decay)
        
        print(f"✓ Optimizer setup with differential learning rates")
        print(f"  Backbone LR: {learning_rate}")
        print(f"  Head LR: {learning_rate * 10}")
        
        return optimizer
    
    def fine_tune_strategy_1(self, train_loader, val_loader, num_epochs=30):
        """
        Strategy 1: Freeze backbone, only train head
        
        Best for limited data
        """
        print(f"\n{'='*70}")
        print("FINE-TUNING STRATEGY 1: Freeze Backbone, Train Head Only")
        print(f"{'='*70}\n")
        
        # Freeze backbone
        self.freeze_backbone()
        
        # Setup optimizer
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        self._train_loop(optimizer, scheduler, train_loader, val_loader, num_epochs)
    
    def fine_tune_strategy_2(self, train_loader, val_loader, num_epochs=30):
        """
        Strategy 2: Differential learning rates
        
        Unfreeze backbone but use lower LR
        """
        print(f"\n{'='*70}")
        print("FINE-TUNING STRATEGY 2: Differential Learning Rates")
        print(f"{'='*70}\n")
        
        # Unfreeze backbone
        self.unfreeze_backbone()
        
        # Setup optimizer with differential LRs
        optimizer = self.setup_optimizer(learning_rate=0.0001)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        self._train_loop(optimizer, scheduler, train_loader, val_loader, num_epochs)
    
    def fine_tune_strategy_3(self, train_loader, val_loader, num_epochs=30):
        """
        Strategy 3: Progressive unfreezing
        
        Start with frozen backbone, then unfreeze and fine-tune
        """
        print(f"\n{'='*70}")
        print("FINE-TUNING STRATEGY 3: Progressive Unfreezing")
        print(f"{'='*70}\n")
        
        # Stage 1: Train head only
        print("\nStage 1: Training head only (epochs 1-10)")
        self.freeze_backbone()
        
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        self._train_loop(optimizer, scheduler, train_loader, val_loader, 10)
        
        # Stage 2: Fine-tune with differential learning rates
        print("\nStage 2: Fine-tuning with differential learning rates (epochs 11-30)")
        self.unfreeze_backbone()
        
        optimizer = self.setup_optimizer(learning_rate=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        self._train_loop(optimizer, scheduler, train_loader, val_loader, 20)
    
    def _train_loop(self, optimizer, scheduler, train_loader, val_loader, num_epochs):
        """Internal training loop"""
        
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            mode="min"
        )
        
        for epoch in range(num_epochs):
            # Train
            self.model.train()
            train_loss = AverageMeter()
            
            for batch_idx, (frames, labels, _) in enumerate(train_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(frames)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss.update(loss.item(), frames.size(0))
            
            # Validate
            self.model.eval()
            val_loss = AverageMeter()
            
            with torch.no_grad():
                for frames, labels, _ in val_loader:
                    frames = frames.to(self.device)
                    labels = labels.to(self.device)
                    
                    logits = self.model(frames)
                    loss = criterion(logits, labels)
                    val_loss.update(loss.item(), frames.size(0))
            
            # Update LR
            scheduler.step()
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss.avg:.4f} | "
                  f"Val Loss: {val_loss.avg:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best
            if val_loss.avg < self.best_val_loss:
                self.best_val_loss = val_loss.avg
                best_path = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_finetuned_best.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ Best model saved: {best_path}")
            
            # Early stopping
            if early_stopping(val_loss.avg):
                print(f"\n⚠ Early stopping triggered")
                break


def main():
    """Main fine-tuning function"""
    
    print("\n" + "="*70)
    print("HRI30 VIDEO ACTION RECOGNITION - FINE-TUNING")
    print("="*70)
    
    # Setup
    device = setup_device(use_gpu=USE_GPU, gpu_id=GPU_ID)
    set_seed(42)
    
    # Load pre-trained model
    print("\nLoading pre-trained model...")
    model = SlowOnlyModel(pretrained="kinetics400")
    model = model.to(device)
    
    best_path = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"✓ Model loaded: {best_path}")
    else:
        print("Note: Using fresh model (best checkpoint not found)")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VALIDATION_SPLIT
    )
    
    # Create fine-tuner
    finetuner = FineTuner(model, device)
    
    # Choose fine-tuning strategy
    print("\n" + "="*70)
    print("SELECT FINE-TUNING STRATEGY:")
    print("="*70)
    print("1. Strategy 1: Freeze backbone, train head only (fastest)")
    print("2. Strategy 2: Differential learning rates (balanced)")
    print("3. Strategy 3: Progressive unfreezing (slowest, best results)")
    
    # Use strategy 2 by default
    strategy = 3
    print(f"\nUsing Strategy {strategy} (Differential Learning Rates)\n")
    
    if strategy == 1:
        finetuner.fine_tune_strategy_1(train_loader, val_loader, num_epochs=30)
    elif strategy == 2:
        finetuner.fine_tune_strategy_2(train_loader, val_loader, num_epochs=30)
    elif strategy == 3:
        finetuner.fine_tune_strategy_3(train_loader, val_loader, num_epochs=30)
    
    print(f"\n{'='*70}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"Fine-tuned model: {CHECKPOINT_DIR / f'{CHECKPOINT_PREFIX}_finetuned_best.pt'}")


if __name__ == "__main__":
    main()
