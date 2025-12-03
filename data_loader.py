# Data loader for HRI30 videos

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import (
    CLASS_NAMES,
    COLOR_JITTER,
    COLOR_JITTER_PARAMS,
    EVAL_RESIZE,
    INPUT_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    RANDOM_CROP,
    RANDOM_FLIP,
    RANDOM_RESIZED_CROP_RATIO,
    RANDOM_RESIZED_CROP_SCALE,
    TARGET_FRAMES,
    TEST_DIR,
    TRAIN_DIR,
    VIDEO_EXTENSION,
    SEED,
)
from utils import get_all_video_files, parse_hri30_filename


class HRI30VideoDataset(Dataset):
    """
    PyTorch Dataset for HRI30 videos
    
    Loads videos, samples frames, and applies augmentation
    """
    
    def __init__(
        self,
        video_files,
        video_dir,
        num_frames=TARGET_FRAMES,
        frame_size=INPUT_SIZE,
        is_train=True,
        has_labels=True,
    ):
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
        self.eval_resize = EVAL_RESIZE

        # Pre-build transforms so we can re-use the same parameters across frames
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        self.color_jitter = (
            transforms.ColorJitter(**COLOR_JITTER_PARAMS) if COLOR_JITTER else None
        )
    
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
        video_path = Path(self.video_dir) / video_filename
        
        # Extract frames
        frames = self._load_video_frames(str(video_path))
        
        # Apply augmentation
        frames = self._apply_transforms(frames)
        
        # Get label if available
        if self.has_labels:
            class_id, _, _ = parse_hri30_filename(video_filename)
            if class_id is None:
                raise ValueError(f"Filename does not match expected pattern: {video_filename}")
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
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")
        
        # Sample frame indices uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frame_index_set = set(frame_indices.tolist())
        
        frames = []
        cap_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if cap_idx in frame_index_set:
                # Convert BGR to RGB for consistency with torchvision
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap_idx += 1
            if cap_idx > frame_indices[-1] and len(frames) >= self.num_frames:
                break
        
        cap.release()
        
        # Ensure we have exactly num_frames (pad if necessary)
        if not frames:
            raise RuntimeError(f"No frames decoded for video: {video_path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # Repeat last frame
        
        frames = frames[:self.num_frames]
        
        return np.stack(frames)  # (T, H, W, 3)
    
    def _apply_transforms(self, frames):
        """
        Apply transforms to frames
        
        Input: frames (T, H, W, 3) as numpy array
        Output: frames (T, 3, H, W) as torch tensor
        """
        pil_frames = [Image.fromarray(frame) for frame in frames]
        transformed_frames = []

        if self.is_train:
            if RANDOM_CROP:
                # Use shared crop params for temporal consistency
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    pil_frames[0],
                    scale=RANDOM_RESIZED_CROP_SCALE,
                    ratio=RANDOM_RESIZED_CROP_RATIO,
                )
            else:
                i = j = 0
                h, w = self.frame_size

            flip = RANDOM_FLIP and random.random() < 0.5
            jitter_fn = None
            if self.color_jitter is not None:
                jitter_fn = transforms.ColorJitter.get_params(
                    self.color_jitter.brightness,
                    self.color_jitter.contrast,
                    self.color_jitter.saturation,
                    self.color_jitter.hue,
                )

            for frame in pil_frames:
                if RANDOM_CROP:
                    frame = TF.resized_crop(frame, i, j, h, w, self.frame_size)
                else:
                    frame = TF.resize(frame, self.frame_size)

                if flip:
                    frame = TF.hflip(frame)
                if jitter_fn is not None:
                    frame = jitter_fn(frame)

                frame = self.to_tensor(frame)
                frame = self.normalize(frame)
                transformed_frames.append(frame)
        else:
            for frame in pil_frames:
                frame = TF.resize(frame, self.eval_resize)
                frame = TF.center_crop(frame, self.frame_size)
                frame = self.to_tensor(frame)
                frame = self.normalize(frame)
                transformed_frames.append(frame)

        return torch.stack(transformed_frames)


def create_train_val_split(train_video_files, val_split=0.1, seed=SEED):
    """
    Split training data into train and validation
    
    Returns:
        train_files, val_files
    """
    if len(train_video_files) == 0:
        return [], []

    labels = []
    for filename in train_video_files:
        class_id, _, _ = parse_hri30_filename(filename)
        if class_id is None:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
        labels.append(class_id)

    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_split, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(train_video_files, labels))

        train_files = [train_video_files[i] for i in train_idx]
        val_files = [train_video_files[i] for i in val_idx]
    except ValueError as err:
        print(f"⚠ Stratified split failed ({err}). Falling back to random split.")
        random.seed(seed)
        np.random.seed(seed)
        shuffled_files = train_video_files.copy()
        random.shuffle(shuffled_files)
        split_idx = int(len(shuffled_files) * (1 - val_split))
        train_files = shuffled_files[:split_idx]
        val_files = shuffled_files[split_idx:]

    return train_files, val_files


def create_dataloaders(batch_size=16, num_workers=4, val_split=0.1, seed=SEED):
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get video files
    train_videos = get_all_video_files(TRAIN_DIR, extension=VIDEO_EXTENSION)
    test_videos = get_all_video_files(TEST_DIR, extension=VIDEO_EXTENSION)
    
    print(f"\nDataLoader Setup:")
    print(f"  Training videos found: {len(train_videos)}")
    print(f"  Test videos found: {len(test_videos)}")
    
    # Split training data
    train_files, val_files = create_train_val_split(train_videos, val_split=val_split, seed=seed)
    
    print(f"  Train split: {len(train_files)}")
    print(f"  Validation split: {len(val_files)}")

    def _count_by_class(files):
        counts = {i: 0 for i in range(1, NUM_CLASSES + 1)}
        for filename in files:
            class_id, _, _ = parse_hri30_filename(filename)
            if class_id is not None and class_id in counts:
                counts[class_id] += 1
        return counts

    train_counts = _count_by_class(train_files)
    val_counts = _count_by_class(val_files)

    print("  Per-class train counts:")
    for cid in sorted(train_counts):
        print(f"    CID{cid:02d} ({CLASS_NAMES[cid][:22]:<22}): {train_counts[cid]}")
    print("  Per-class val counts:")
    for cid in sorted(val_counts):
        print(f"    CID{cid:02d} ({CLASS_NAMES[cid][:22]:<22}): {val_counts[cid]}")
    
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
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
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
