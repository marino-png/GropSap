# Evaluation and prediction script

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, CHECKPOINT_PREFIX, NUM_CLASSES,
    CLASS_NAMES, TOP_K_PREDICTIONS, USE_GPU, GPU_ID, BATCH_SIZE,
    NUM_WORKERS, CONFUSION_MATRIX, SAVE_PLOTS, MODEL_TYPE, BACKBONE,
    PRETRAINED_DATASET, HEAD_DROPOUT, FREEZE_BACKBONE, VALIDATION_SPLIT, SEED
)

from data_loader import create_dataloaders
from model import build_model
from utils import setup_device, plot_confusion_matrix, save_predictions_to_csv, AverageMeter, set_seed


class Evaluator:
    """Evaluator class for testing and prediction"""
    
    def __init__(self, model, device):
        """
        Args:
            model: PyTorch model
            device: torch.device
        """
        self.model = model
        self.device = device
        self.model.eval()

    @staticmethod
    def _topk_accuracy(logits, labels, k=5):
        """Compute top-k accuracy for a batch."""
        _, pred = logits.topk(k, 1, True, True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        correct_total = correct.float().sum().item()
        return correct_total / labels.size(0)
    
    def predict_batch(self, frames):
        """
        Predict on a batch of frames
        
        Args:
            frames: (B, T, 3, H, W)
        
        Returns:
            logits: (B, num_classes)
            predictions: (B,) - predicted class indices
            confidences: (B,) - confidence scores
            top_k: list of (class_idx, class_name, confidence) tuples
        """
        with torch.no_grad():
            logits = self.model(frames)
            probabilities = torch.softmax(logits, dim=1)
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        confidences = torch.max(probabilities, dim=1).values.cpu().numpy()
        
        # Top-K predictions
        k = min(TOP_K_PREDICTIONS, NUM_CLASSES)
        top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
        
        top_k_list = []
        for b in range(len(frames)):
            batch_top_k = []
            for k_idx in range(top_k_indices.shape[1]):
                class_idx = top_k_indices[b, k_idx].item() + 1  # Convert to 1-indexed
                class_name = CLASS_NAMES.get(class_idx, f"Unknown_{class_idx}")
                prob = top_k_probs[b, k_idx].item()
                batch_top_k.append((class_idx, class_name, prob))
            top_k_list.append(batch_top_k)
        
        return predictions, confidences, top_k_list
    
    def evaluate(self, val_loader):
        """
        Evaluate on validation set
        
        Returns: dict with metrics
        """
        print(f"\n{'='*70}")
        print("EVALUATION ON VALIDATION SET")
        print(f"{'='*70}\n")
        
        all_preds = []
        all_labels = []
        total_acc = AverageMeter()
        total_top5 = AverageMeter()
        
        with torch.no_grad():
            for batch_idx, (frames, labels, filenames) in enumerate(val_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Predict
                logits = self.model(frames)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Compute accuracy
                acc = accuracy_score(labels_np, preds)
                top5 = self._topk_accuracy(logits, labels, k=min(TOP_K_PREDICTIONS, NUM_CLASSES))
                total_acc.update(acc, len(frames))
                total_top5.update(top5, len(frames))
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch [{batch_idx+1}/{len(val_loader)}] Acc@1: {acc:.4f} | Acc@5: {top5:.4f}")
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = {
            "top_1_accuracy": accuracy_score(all_labels, all_preds) * 100,
            "top_5_accuracy": total_top5.avg * 100,
        }
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_preds,
            labels=range(NUM_CLASSES),
            target_names=[CLASS_NAMES[i+1] for i in range(NUM_CLASSES)],
            output_dict=True
        )
        
        print(f"\nOverall Accuracy: {metrics['top_1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy:   {metrics['top_5_accuracy']:.2f}%")
        print(f"\nPer-class Results:")
        print("-" * 70)
        print(f"{'Class':<40} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for i in range(NUM_CLASSES):
            class_name = CLASS_NAMES[i+1]
            if class_name in class_report:
                precision = class_report[class_name]['precision']
                recall = class_report[class_name]['recall']
                f1 = class_report[class_name]['f1-score']
                print(f"{class_name:<40} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        # Confusion matrix
        if CONFUSION_MATRIX:
            save_path = RESULTS_DIR / "confusion_matrix.png" if SAVE_PLOTS else None
            plot_confusion_matrix(
                all_labels, all_preds, CLASS_NAMES,
                save_path=save_path,
                title="Confusion Matrix - Validation Set"
            )
        
        return metrics, all_labels, all_preds
    
    def predict_on_test(self, test_loader):
        """
        Make predictions on test set
        
        Returns: list of result dicts
        """
        print(f"\n{'='*70}")
        print("PREDICTION ON TEST SET")
        print(f"{'='*70}\n")
        
        results = []
        
        with torch.no_grad():
            for batch_idx, (frames, _, filenames) in enumerate(test_loader):
                frames = frames.to(self.device)
                
                # Predict
                preds, confidences, top_k_list = self.predict_batch(frames)
                
                # Collect results
                for i, filename in enumerate(filenames):
                    pred_class = preds[i] + 1  # Convert to 1-indexed
                    pred_label = CLASS_NAMES.get(pred_class, f"Unknown_{pred_class}")
                    
                    result = {
                        "video_name": filename,
                        "predicted_class": pred_class,
                        "predicted_label": pred_label,
                        "confidence": confidences[i],
                        "top_5": top_k_list[i]
                    }
                    results.append(result)

                    # Print per-video summary with top-5 confidences
                    print(f"Video: {filename}")
                    print(f"  Top-1: {pred_label} ({confidences[i]:.4f})")
                    print("  Top-5:")
                    for rank, (cls_idx, cls_name, prob) in enumerate(top_k_list[i], start=1):
                        print(f"    {rank}. {cls_name} ({prob:.4f})")
                    print("-" * 70)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch [{batch_idx+1}/{len(test_loader)}] - Predictions made")
        
        print(f"\n✓ Total predictions: {len(results)}")
        
        # Save results
        save_path = RESULTS_DIR / "test_predictions.csv"
        save_predictions_to_csv(results, save_path)
        
        return results


def main():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("HRI30 VIDEO ACTION RECOGNITION - EVALUATION")
    print("="*70)
    set_seed(SEED)
    
    # Setup device
    device = setup_device(use_gpu=USE_GPU, gpu_id=GPU_ID)
    
    # Load model
    print("\nLoading model...")
    model = build_model(
        model_type=MODEL_TYPE,
        backbone=BACKBONE,
        pretrained=PRETRAINED_DATASET,
        dropout_rate=HEAD_DROPOUT,
        freeze_backbone=FREEZE_BACKBONE,
    )
    model = model.to(device)
    
    best_model_path = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_best.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✓ Model loaded: {best_model_path}")
    else:
        print(f"⚠ Best model not found at {best_model_path}")
        print("  Using untrained model for testing purposes")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VALIDATION_SPLIT,
    )
    
    # Create evaluator
    evaluator = Evaluator(model, device)
    
    # Evaluate on validation set
    val_metrics, val_labels, val_preds = evaluator.evaluate(val_loader)
    
    # Predict on test set
    test_results = evaluator.predict_on_test(test_loader)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"  - Predictions: test_predictions.csv")
    print(f"  - Confusion Matrix: confusion_matrix.png")


if __name__ == "__main__":
    main()
