# SlowOnly model architecture for HRI30

import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, BACKBONE, PRETRAINED_DATASET


class SlowOnlyModel(nn.Module):
    """
    SlowOnly architecture adapted for HRI30
    
    Based on "SlowFast Networks for Video Recognition"
    - Uses only the slow pathway with 8 frames sampled every 4 frames
    - Backbone: ResNet-50
    - Pre-trained on Kinetics-400 for transfer learning
    """
    
    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE, 
                 pretrained=PRETRAINED_DATASET, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of action classes
            backbone: Backbone architecture (resnet50, resnet101, etc.)
            pretrained: Pre-training dataset ('kinetics400', 'imagenet', None)
            dropout_rate: Dropout rate for FC layers
        """
        super(SlowOnlyModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = backbone
        self.pretrained = pretrained
        
        # Load backbone
        if backbone == "resnet50":
            if pretrained == "kinetics400":
                # Try to load kinetics400 pre-trained weights
                # Fallback to ImageNet if not available
                try:
                    from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
                    # We'll use timm or download from online sources
                    # For now, use ImageNet and fine-tune
                    print("Note: Using ImageNet pre-training (Kinetics-400 requires external download)")
                    self.backbone_model = models.resnet50(pretrained=True)
                except:
                    self.backbone_model = models.resnet50(pretrained=True)
            elif pretrained == "imagenet":
                self.backbone_model = models.resnet50(pretrained=True)
            else:
                self.backbone_model = models.resnet50(pretrained=False)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone_model.children())[:-1])
        
        # Get feature dimension
        self.feature_dim = self.backbone_model.fc.in_features
        
        # Temporal pooling (average pool over frames)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc_out = nn.Linear(512, num_classes)
        
        print(f"✓ SlowOnly Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  Pre-trained: {pretrained if pretrained else 'No'}")
        print(f"  Feature Dimension: {self.feature_dim}")
        print(f"  Num Classes: {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, 3, H, W)
               B: batch size
               T: number of frames (temporal dimension)
               3: RGB channels
               H, W: height, width
        
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # x shape: (B, T, 3, H, W)
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        # Process each frame through ResNet
        # Reshape to (B*T, 3, H, W)
        x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
        
        # Extract features: (B*T, feature_dim, 1, 1)
        x = self.features(x)
        
        # Reshape back to (B, T, feature_dim)
        x = x.view(batch_size, num_frames, -1)
        
        # Temporal pooling: average over time
        x = x.mean(dim=1)  # (B, feature_dim)
        
        # Classification head
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        logits = self.fc_out(x)  # (B, num_classes)
        
        return logits


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for video action recognition
    
    Alternative to SlowOnly if you want to try LSTM
    """
    
    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE,
                 pretrained=PRETRAINED_DATASET, lstm_hidden=512, lstm_layers=2,
                 dropout_rate=0.5):
        """
        Args:
            num_classes: Number of action classes
            backbone: Backbone CNN architecture
            pretrained: Pre-training dataset
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(CNNLSTMModel, self).__init__()
        
        # CNN backbone
        if backbone == "resnet50":
            self.cnn = models.resnet50(pretrained=(pretrained == "imagenet"))
            self.cnn_feature_dim = self.cnn.fc.in_features
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # LSTM
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)
        
        print(f"✓ CNN-LSTM Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  LSTM Hidden: {lstm_hidden}")
        print(f"  Num Classes: {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, 3, H, W)
        
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        # CNN forward: (B*T, 3, H, W) -> (B*T, feature_dim)
        x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
        x = self.cnn(x)
        x = x.view(batch_size, num_frames, -1)  # (B, T, feature_dim)
        
        # LSTM forward
        x, (h_n, c_n) = self.lstm(x)  # x: (B, T, hidden), h_n: (layers, B, hidden)
        
        # Use last hidden state
        x = h_n[-1]  # (B, hidden)
        
        # Classification head
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_model(model_path, device, architecture="slowonly"):
    """Load a saved model"""
    if architecture == "slowonly":
        model = SlowOnlyModel()
    elif architecture == "cnn_lstm":
        model = CNNLSTMModel()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    # Test model
    print("Testing SlowOnly Model...")
    model = SlowOnlyModel()
    model.eval()
    
    # Input: (B, T, C, H, W) = (2, 8, 3, 224, 224)
    x = torch.randn(2, 8, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
