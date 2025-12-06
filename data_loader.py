# Data loader for HRI30 videos

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random

from config import (
    TRAIN_DIR, TEST_DIR, TARGET_FRAMES, INPUT_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, RANDOM_FLIP, RANDOM_CROP,
    CROP_SCALE, COLOR_JITTER, COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST, COLOR_JITTER_SATURATION,
    NUM_CLASSES, CLASS_NAMES
)
from utils import parse_hri30_filename, get_all_video_files


class HRI30VideoDataset(Dataset):
    """
    PyTorch Dataset for HRI30 videos
    
    Loads videos, samples frames, and applies augmentation
    """
    
    def __init__(self, video_files, video_dir, num_frames=TARGET_FRAMES, 
                 frame_size=INPUT_SIZE, is_train=True, has_labels=True):
        """
        Args:
            video_files: List of video filenames
            video_dir: Directory containing videos
            num_frames: Number of frames to sample per video
            frame_size: Target frame size (height, width)
            is_train: Whether this is training data (affects augmentation)
            has_labels: Whether videos have labels in filename
        """
        self.video_files = video_files
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.is_train = is_train
        self.has_labels = has_labels
        
        # Augmentation transforms
        if is_train:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ])
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            frames: torch.Tensor of shape (T, 3, H, W) - T frames
            label: int - class index (0-29) or -1 if no label
            filename: str - video filename
        """
        video_filename = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_filename)
        
        # Extract frames
        frames = self._load_video_frames(video_path)
        
        # Apply augmentation
        frames = self._apply_transforms(frames)
        
        # Get label if available
        if self.has_labels:
            class_id, _, _ = parse_hri30_filename(video_filename)
            label = class_id - 1  # Convert to 0-indexed
        else:
            label = -1
        
        return frames, label, video_filename
    
    def _load_video_frames(self, video_path):
        """
        Load video and extract frames
        
        Returns:
            frames: np.array of shape (T, H, W, 3) in RGB
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices uniformly
        if total_frames <= self.num_frames:
            # If video has fewer frames than target, take all
            frame_indices = list(range(total_frames))
        else:
            # Uniformly sample num_frames from video
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        frame_idx = 0
        cap_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if cap_idx in frame_indices:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame = cv2.resize(frame, self.frame_size)
                
                # Add random augmentation during training
                if self.is_train:
                    frame = self._augment_frame(frame)
                
                frames.append(frame)
            
            cap_idx += 1
        
        cap.release()
        
        # Ensure we have exactly num_frames (pad if necessary)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # Repeat last frame
        
        frames = frames[:self.num_frames]
        
        return np.stack(frames)  # (T, H, W, 3)
    
    def _augment_frame(self, frame):
        """Apply random augmentation to frame"""
        # Random horizontal flip (disabled for action recognition)
        if RANDOM_FLIP and random.random() < 0.5:
            frame = cv2.flip(frame, 1)
        
        # Random crop
        if RANDOM_CROP and random.random() < 0.5:
            h, w = frame.shape[:2]
            scale = random.uniform(CROP_SCALE[0], CROP_SCALE[1])
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Random crop position
            y = random.randint(0, max(0, h - new_h))
            x = random.randint(0, max(0, w - new_w))
            
            frame = frame[y:y+new_h, x:x+new_w]
            frame = cv2.resize(frame, self.frame_size)
        
        # Color jitter (brightness, contrast, saturation)
        if COLOR_JITTER:
            # Brightness
            if random.random() < 0.5:
                brightness_factor = random.uniform(1 - COLOR_JITTER_BRIGHTNESS, 1 + COLOR_JITTER_BRIGHTNESS)
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness_factor - 1) * 127)
            
            # Contrast
            if random.random() < 0.5:
                contrast_factor = random.uniform(1 - COLOR_JITTER_CONTRAST, 1 + COLOR_JITTER_CONTRAST)
                frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
            
            # Saturation (convert to HSV, modify, convert back)
            if random.random() < 0.5:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                saturation_factor = random.uniform(1 - COLOR_JITTER_SATURATION, 1 + COLOR_JITTER_SATURATION)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Random brightness/contrast (legacy, keep for compatibility)
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
        
        return frame
    
    def _apply_transforms(self, frames):
        """
        Apply transforms to frames
        
        Input: frames (T, H, W, 3) as numpy array
        Output: frames (T, 3, H, W) as torch tensor
        """
        transformed_frames = []
        
        for frame in frames:
            # PIL Image
            pil_frame = transforms.ToPILImage()(frame)
            
            # Apply transforms
            tensor_frame = self.transforms(pil_frame)
            transformed_frames.append(tensor_frame)
        
        # Stack into (T, 3, H, W)
        return torch.stack(transformed_frames)


def create_train_val_split(train_video_files, val_split=0.1, seed=42):
    """
    Split training data into train and validation
    
    Returns:
        train_files, val_files
    """
    random.seed(seed)
    np.random.seed(seed)
    
    shuffled_files = train_video_files.copy()
    random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * (1 - val_split))
    
    return shuffled_files[:split_idx], shuffled_files[split_idx:]


def create_dataloaders(batch_size=16, num_workers=4, val_split=0.1):
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get video files
    train_videos = get_all_video_files(TRAIN_DIR)
    test_videos = get_all_video_files(TEST_DIR)
    
    print(f"\nDataLoader Setup:")
    print(f"  Training videos found: {len(train_videos)}")
    print(f"  Test videos found: {len(test_videos)}")
    
    # Split training data
    train_files, val_files = create_train_val_split(train_videos, val_split=val_split)
    
    print(f"  Train split: {len(train_files)}")
    print(f"  Validation split: {len(val_files)}")
    
    # Create datasets
    train_dataset = HRI30VideoDataset(
        train_files, TRAIN_DIR, 
        num_frames=TARGET_FRAMES, frame_size=INPUT_SIZE,
        is_train=True, has_labels=True
    )
    
    val_dataset = HRI30VideoDataset(
        val_files, TRAIN_DIR,
        num_frames=TARGET_FRAMES, frame_size=INPUT_SIZE,
        is_train=False, has_labels=True
    )
    
    test_dataset = HRI30VideoDataset(
        test_videos, TEST_DIR,
        num_frames=TARGET_FRAMES, frame_size=INPUT_SIZE,
        is_train=False, has_labels=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    print("Testing HRI30 DataLoader...")
    
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4, num_workers=0)
    
    # Check a batch
    for frames, labels, filenames in train_loader:
        print(f"Batch shape: {frames.shape}")  # Should be (B, T, 3, H, W)
        print(f"Labels: {labels}")
        print(f"Filenames: {filenames}")
        print(f"Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
        break
