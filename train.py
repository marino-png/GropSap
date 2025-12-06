# Training script for HRI30

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from pathlib import Path

from config import (
    BATCH_SIZE, NUM_EPOCHS, INITIAL_LR, LR_SCHEDULER, LR_STEP_SIZE, LR_GAMMA,
    LR_WARMUP_EPOCHS, OPTIMIZER, MOMENTUM, WEIGHT_DECAY, USE_GPU, GPU_ID, 
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, CHECKPOINT_DIR, RESULTS_DIR, 
    LOGS_DIR, CHECKPOINT_PREFIX, LOG_INTERVAL, SAVE_CHECKPOINT_INTERVAL, SEED, 
    CUDNN_DETERMINISTIC, CUDNN_BENCHMARK, NUM_WORKERS, PLOT_LOSS, SAVE_PLOTS,
    VALIDATION_SPLIT, GRADIENT_ACCUMULATION_STEPS, USE_MIXED_PRECISION,
    CLEAR_CACHE_EVERY_N_BATCHES
)

from data_loader import create_dataloaders
from model import SlowOnlyModel, CNNLSTMModel
from utils import (
    setup_device, set_seed, EarlyStopping, AverageMeter, 
    plot_training_history, print_metrics
)


class Trainer:
    """Trainer class for HRI30 model"""
    
    def __init__(self, model, device, config=None):
        """
        Args:
            model: PyTorch model
            device: torch.device
            config: Configuration dict (optional)
        """
        self.model = model
        self.device = device
        self.config = config or {}
        
        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        
        print(f"\n{'='*70}")
        print("TRAINER INITIALIZED")
        print(f"{'='*70}")
    
    def setup_optimizer_and_scheduler(self, learning_rate=INITIAL_LR, optimizer_name=OPTIMIZER):
        """Setup optimizer and learning rate scheduler"""
        
        if optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                nesterov=True
            )
            print(f"✓ Optimizer: SGD (momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
        
        elif optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY
            )
            print(f"✓ Optimizer: Adam (weight_decay={WEIGHT_DECAY})")
        
        # Warmup epochs
        self.warmup_epochs = LR_WARMUP_EPOCHS if LR_WARMUP_EPOCHS > 0 else 0
        if self.warmup_epochs > 0:
            print(f"✓ LR Warmup: {self.warmup_epochs} epochs")
        
        # Learning rate scheduler (starts after warmup)
        if LR_SCHEDULER == "cosine":
            # Adjust T_max to account for warmup
            effective_epochs = NUM_EPOCHS - self.warmup_epochs
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=effective_epochs, eta_min=1e-5)
            print(f"✓ LR Scheduler: Cosine Annealing (T_max={effective_epochs}, eta_min=1e-5)")
        
        elif LR_SCHEDULER == "step":
            self.scheduler = StepLR(self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
            print(f"✓ LR Scheduler: Step (step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")
        
        elif LR_SCHEDULER == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=LR_GAMMA)
            print(f"✓ LR Scheduler: Exponential (gamma={LR_GAMMA})")
        
        else:
            self.scheduler = None
            print(f"✓ No LR Scheduler")
    
    def train_epoch(self, train_loader, criterion, scaler=None):
        """Train for one epoch with gradient accumulation and mixed precision"""
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        # Zero gradients at start of epoch
        self.optimizer.zero_grad()
        
        for batch_idx, (frames, labels, filenames) in enumerate(train_loader):
            # Move to device
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = self.model(frames)
                    loss = criterion(logits, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
            else:
                # Standard precision
                logits = self.model(frames)
                loss = criterion(logits, labels)
                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
            
            # Compute accuracy (use full loss for logging)
            _, preds = torch.max(logits, 1)
            acc = (preds == labels).float().mean()
            
            # Update metrics (multiply by accumulation steps to get true loss)
            losses.update(loss.item() * GRADIENT_ACCUMULATION_STEPS, frames.size(0))
            accs.update(acc.item(), frames.size(0))
            
            # Gradient accumulation: only step optimizer every N batches
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping and monitoring
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Log gradient norm occasionally for debugging
                if (batch_idx + 1) % (LOG_INTERVAL * 5) == 0:
                    print(f"    Gradient Norm: {grad_norm:.4f}")
                
                self.optimizer.zero_grad()
            
            # Clear cache periodically to prevent memory fragmentation
            if CLEAR_CACHE_EVERY_N_BATCHES > 0 and (batch_idx + 1) % CLEAR_CACHE_EVERY_N_BATCHES == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {losses.avg:.4f} | Acc: {accs.avg:.4f}")
        
        # Handle remaining gradients if batch count is not divisible by accumulation steps
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            if scaler is not None:
                scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return losses.avg, accs.avg
    
    def validate(self, val_loader, criterion):
        """Validate on validation set"""
        self.model.eval()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        with torch.no_grad():
            for frames, labels, filenames in val_loader:
                # Move to device
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(frames)
                loss = criterion(logits, labels)
                
                # Compute accuracy
                _, preds = torch.max(logits, 1)
                acc = (preds == labels).float().mean()
                
                # Update metrics
                losses.update(loss.item(), frames.size(0))
                accs.update(acc.item(), frames.size(0))
        
        return losses.avg, accs.avg
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_name = f"{CHECKPOINT_PREFIX}_epoch{epoch:03d}.pt"
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        
        torch.save(self.model.state_dict(), checkpoint_path)
        
        if is_best:
            best_path = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_best.pt"
            torch.save(self.model.state_dict(), best_path)
            print(f"✓ Best checkpoint saved: {best_path}")
        else:
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, use_class_weights=False):
        """
        Train the model
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            use_class_weights: Whether to use class weights for imbalanced data
        """
        # Compute class weights if requested
        class_weights = None
        if use_class_weights:
            try:
                # Collect all labels from training set
                all_labels = []
                for _, labels, _ in train_loader:
                    all_labels.extend(labels.cpu().numpy())
                
                # Compute balanced class weights
                unique_labels = np.unique(all_labels)
                weights = compute_class_weight('balanced', classes=unique_labels, y=all_labels)
                class_weights = torch.FloatTensor(weights).to(self.device)
                print(f"✓ Using class weights: {class_weights}")
            except (ImportError, NameError):
                print("⚠ sklearn not available, skipping class weights")
        
        # Use label smoothing if configured
        from config import LABEL_SMOOTHING
        if LABEL_SMOOTHING > 0:
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            print(f"✓ Using Label Smoothing: {LABEL_SMOOTHING}")
        else:
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Mixed precision training
        scaler = None
        if USE_MIXED_PRECISION and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            print(f"✓ Mixed Precision Training: Enabled (FP16)")
        else:
            print(f"✓ Mixed Precision Training: Disabled (FP32)")
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            mode="min"
        )
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print("TRAINING START")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"Total batches per epoch: {len(train_loader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, scaler=scaler)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate (warmup first, then main scheduler)
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                # Warmup: linearly increase LR from 0 to INITIAL_LR
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = INITIAL_LR * warmup_factor
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc * 100)  # Convert to percentage
            self.val_accs.append(val_acc * 100)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(epoch + 1)
            
            # Check if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch + 1, is_best=True)
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch: {self.best_epoch}")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Best epoch: {self.best_epoch} (Val Loss: {self.best_val_loss:.4f})")
        print(f"{'='*70}\n")
        
        # Plot results
        if PLOT_LOSS:
            save_path = RESULTS_DIR / "training_curves.png" if SAVE_PLOTS else None
            plot_training_history(
                self.train_losses, self.val_losses,
                self.train_accs, self.val_accs,
                save_path=save_path,
                title="Training History - HRI30"
            )
    
    def load_best_model(self):
        """Load the best checkpoint"""
        best_path = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_best.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"✓ Best model loaded: {best_path}")
        else:
            print(f"⚠ Best model not found at {best_path}")


def main():
    """Main training function"""
    
    # Setup
    print("\n" + "="*70)
    print("HRI30 VIDEO ACTION RECOGNITION - SLOWONLY")
    print("="*70)
    
    set_seed(SEED)
    
    # CUDA setup
    if USE_GPU:
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(f"cuda:{GPU_ID}")
                torch.cuda.set_device(device)
                print(f"✓ CUDA device set: {torch.cuda.get_device_name(GPU_ID)}")
            except Exception as e:
                print(f"⚠ Error setting CUDA device: {e}")
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    device = setup_device(use_gpu=USE_GPU, gpu_id=GPU_ID)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VALIDATION_SPLIT
    )
    
    # Create model with improved architecture
    print("\nInitializing model...")
    from config import LABEL_SMOOTHING
    model = SlowOnlyModel(use_temporal=True, dropout_rate=0.6)  # Enable temporal modeling
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer and train
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Load best model for testing
    trainer.load_best_model()
    
    print("\n✓ Training complete! Ready for evaluation.")
    print(f"  Best model: {CHECKPOINT_DIR / f'{CHECKPOINT_PREFIX}_best.pt'}")
    print(f"  Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
