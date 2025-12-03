# Training script for HRI30

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

from config import (
    BATCH_SIZE, NUM_EPOCHS, INITIAL_LR, LR_SCHEDULER, LR_STEP_SIZE, LR_GAMMA,
    OPTIMIZER, MOMENTUM, WEIGHT_DECAY, USE_GPU, GPU_ID, EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR,
    CHECKPOINT_PREFIX, LOG_INTERVAL, SAVE_CHECKPOINT_INTERVAL, SEED, 
    CUDNN_DETERMINISTIC, CUDNN_BENCHMARK, NUM_WORKERS, PLOT_LOSS, SAVE_PLOTS,
    VALIDATION_SPLIT, MODEL_TYPE, BACKBONE, PRETRAINED_DATASET, HEAD_DROPOUT,
    FREEZE_BACKBONE, LABEL_SMOOTHING
)

from data_loader import create_dataloaders
from model import build_model
from utils import (
    setup_device, set_seed, EarlyStopping, AverageMeter, 
    plot_training_history, print_metrics
)


def accuracy_topk(logits, targets, topk=(1,)):
    """Compute top-k accuracies for given logits/targets."""
    max_k = max(topk)
    _, pred = logits.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / targets.size(0)).item())
    return res


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
        self.train_top5 = []
        self.val_top5 = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        
        print(f"\n{'='*70}")
        print("TRAINER INITIALIZED")
        print(f"{'='*70}")
    
    def setup_optimizer_and_scheduler(self, learning_rate=INITIAL_LR, optimizer_name=OPTIMIZER):
        """Setup optimizer and learning rate scheduler"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("No trainable parameters found. Check freeze_backbone setting.")

        if optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=learning_rate,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                nesterov=True
            )
            print(f"✓ Optimizer: SGD (momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
        
        elif optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY
            )
            print(f"✓ Optimizer: Adam (weight_decay={WEIGHT_DECAY})")
        
        # Learning rate scheduler
        if LR_SCHEDULER == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
            print(f"✓ LR Scheduler: Cosine Annealing (T_max={NUM_EPOCHS})")
        
        elif LR_SCHEDULER == "step":
            self.scheduler = StepLR(self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
            print(f"✓ LR Scheduler: Step (step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")
        
        elif LR_SCHEDULER == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=LR_GAMMA)
            print(f"✓ LR Scheduler: Exponential (gamma={LR_GAMMA})")
        
        else:
            self.scheduler = None
            print(f"✓ No LR Scheduler")
    
    def train_epoch(self, train_loader, criterion):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        top5_meter = AverageMeter()
        
        for batch_idx, (frames, labels, filenames) in enumerate(train_loader):
            # Move to device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy
            top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))
            
            # Update metrics
            losses.update(loss.item(), frames.size(0))
            accs.update(top1, frames.size(0))
            top5_meter.update(top5, frames.size(0))
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {losses.avg:.4f} | Acc@1: {accs.avg:.4f} | Acc@5: {top5_meter.avg:.4f}")
        
        return losses.avg, accs.avg, top5_meter.avg
    
    def validate(self, val_loader, criterion):
        """Validate on validation set"""
        self.model.eval()
        
        losses = AverageMeter()
        accs = AverageMeter()
        top5_meter = AverageMeter()
        
        with torch.no_grad():
            for frames, labels, filenames in val_loader:
                # Move to device
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(frames)
                loss = criterion(logits, labels)
                
                # Compute accuracy
                top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))
                
                # Update metrics
                losses.update(loss.item(), frames.size(0))
                accs.update(top1, frames.size(0))
                top5_meter.update(top5, frames.size(0))
        
        return losses.avg, accs.avg, top5_meter.avg
    
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
        
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS):
        """
        Train the model
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
        """
        if LABEL_SMOOTHING > 0:
            try:
                criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
                print(f"✓ Using CrossEntropyLoss with label_smoothing={LABEL_SMOOTHING}")
            except TypeError:
                print("⚠ Label smoothing not supported, falling back to standard CE.")
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
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
        print(f"Total batches per epoch: {len(train_loader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 70)
            
            # Train
            train_loss, train_acc, train_top5 = self.train_epoch(train_loader, criterion)
            
            # Validate
            val_loss, val_acc, val_top5 = self.validate(val_loader, criterion)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc * 100)  # Convert to percentage
            self.val_accs.append(val_acc * 100)
            self.train_top5.append(train_top5 * 100)
            self.val_top5.append(val_top5 * 100)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Train Loss: {train_loss:.4f} | Train Acc@1: {train_acc:.4f} | Train Acc@5: {train_top5:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc@1:   {val_acc:.4f} | Val Acc@5:   {val_top5:.4f}")
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
    print("HRI30 VIDEO ACTION RECOGNITION - TRAINING")
    print("="*70)
    
    set_seed(SEED)
    
    torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    device = setup_device(use_gpu=USE_GPU, gpu_id=GPU_ID)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VALIDATION_SPLIT
    )
    
    # Create model
    print("\nInitializing model...")
    print(f"Model config: type={MODEL_TYPE}, backbone={BACKBONE}, pretrained={PRETRAINED_DATASET}, "
          f"freeze_backbone={FREEZE_BACKBONE}, dropout={HEAD_DROPOUT}")
    model = build_model(
        model_type=MODEL_TYPE,
        backbone=BACKBONE,
        pretrained=PRETRAINED_DATASET,
        dropout_rate=HEAD_DROPOUT,
        freeze_backbone=FREEZE_BACKBONE,
    )
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
