# SlowOnly model architecture for HRI30

import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, BACKBONE, PRETRAINED_DATASET


class SlowOnlyModel(nn.Module):
    """
    Improved SlowOnly architecture with temporal modeling for HRI30
    
    Key improvements:
    - Added LSTM for temporal modeling (captures action sequences)
    - Reduced classification head size to prevent overfitting
    - Increased dropout for better regularization
    """
    
    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE, 
                 pretrained=PRETRAINED_DATASET, dropout_rate=0.6, use_temporal=True):
        """
        Args:
            num_classes: Number of action classes
            backbone: Backbone architecture (resnet50, resnet101, etc.)
            pretrained: Pre-training dataset ('kinetics400', 'imagenet', None)
            dropout_rate: Dropout rate for FC layers (increased to 0.6)
            use_temporal: Whether to use LSTM for temporal modeling
        """
        super(SlowOnlyModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = backbone
        self.pretrained = pretrained
        self.use_temporal = use_temporal
        
        # Load video pre-trained backbone (much better than ImageNet for video tasks)
        from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
        
        if pretrained == "r2plus1d_18":
            # R(2+1)D - Best for video action recognition, pre-trained on Kinetics-400
            # Use weights parameter instead of deprecated pretrained
            try:
                from torchvision.models.video import R2Plus1D_18_Weights
                self.backbone_model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
            except:
                # Fallback for older torchvision versions
                self.backbone_model = r2plus1d_18(pretrained=True)
            self.feature_dim = 512  # R2+1D-18 output dimension
            self.is_video_model = True
            print("✓ Using R(2+1)D-18 pre-trained on Kinetics-400")
        
        elif pretrained == "r3d_18":
            # R3D - 3D ResNet, pre-trained on Kinetics-400
            try:
                from torchvision.models.video import R3D_18_Weights
                self.backbone_model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            except:
                self.backbone_model = r3d_18(pretrained=True)
            self.feature_dim = 512
            self.is_video_model = True
            print("✓ Using R3D-18 pre-trained on Kinetics-400")
        
        elif pretrained == "mc3_18":
            # Mixed Convolution 3D, pre-trained on Kinetics-400
            try:
                from torchvision.models.video import MC3_18_Weights
                self.backbone_model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            except:
                self.backbone_model = mc3_18(pretrained=True)
            self.feature_dim = 512
            self.is_video_model = True
            print("✓ Using MC3-18 pre-trained on Kinetics-400")
        
        elif pretrained == "imagenet" or backbone == "resnet50":
            # Fallback to ImageNet ResNet (not ideal for video)
            self.backbone_model = models.resnet50(pretrained=(pretrained == "imagenet"))
            self.feature_dim = self.backbone_model.fc.in_features
            self.is_video_model = False
            print("⚠ Using ImageNet ResNet-50 (not optimal for video)")
        
        else:
            # No pre-training
            if backbone == "resnet50":
                self.backbone_model = models.resnet50(pretrained=False)
                self.feature_dim = self.backbone_model.fc.in_features
            else:
                raise ValueError(f"Unknown backbone: {backbone}")
            self.is_video_model = False
            print("⚠ No pre-training (random initialization)")
        
        # Remove the final classification layer
        if self.is_video_model:
            # Video models have different structure - remove fc layer
            # R2+1D, R3D, MC3 have: stem -> layer1-4 -> avgpool -> fc
            # We want everything except fc
            self.features = nn.Sequential(*list(self.backbone_model.children())[:-1])
        else:
            # ResNet structure
            self.features = nn.Sequential(*list(self.backbone_model.children())[:-1])
        
        # Temporal modeling with LSTM (only for non-video models)
        # Video models (R2+1D, R3D) already have temporal modeling built-in
        is_video_pretrained = self.is_video_model
        if use_temporal and not is_video_pretrained:
            self.lstm = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=512,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate if 2 > 1 else 0,
                bidirectional=False
            )
            temporal_output_dim = 512
        else:
            self.lstm = None
            temporal_output_dim = self.feature_dim
        
        # Reduced classification head to prevent overfitting
        # Original: 2048 -> 1024 -> 512 -> 30
        # New: 512 -> 256 -> 30 (or feature_dim -> 256 -> 30 if no LSTM)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(temporal_output_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)
        
        print(f"✓ Improved SlowOnly Model initialized")
        print(f"  Backbone: {backbone if not self.is_video_model else pretrained}")
        print(f"  Pre-trained: {pretrained if pretrained else 'No'}")
        print(f"  Feature Dimension: {self.feature_dim}")
        if self.is_video_model:
            print(f"  Temporal Modeling: Built-in (3D convolutions)")
        else:
            print(f"  Temporal Modeling: {'LSTM' if use_temporal and self.lstm is not None else 'Average Pooling'}")
        print(f"  Dropout Rate: {dropout_rate}")
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
        
        if self.is_video_model:
            # Video models (R2+1D, R3D, MC3) expect (B, C, T, H, W) format
            # Convert from (B, T, C, H, W) to (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
            
            # Extract features using 3D convolutions
            # Video models already have temporal modeling built-in
            x = self.features(x)
            
            # Video models output: (B, 512, 1, 1, 1) - confirmed from test
            # Flatten to (B, 512) - use view for more reliable flattening
            if x.dim() > 2:
                # Flatten all dimensions except batch dimension
                x = x.view(batch_size, -1)
            
            # Verify we got the right feature dimension
            if x.shape[1] != self.feature_dim:
                # If wrong size, try to fix it
                if x.shape[1] > self.feature_dim:
                    # Take first feature_dim elements
                    x = x[:, :self.feature_dim]
                elif x.shape[1] < self.feature_dim:
                    # This shouldn't happen, but pad if needed
                    padding = torch.zeros(batch_size, self.feature_dim - x.shape[1], 
                                         device=x.device, dtype=x.dtype)
                    x = torch.cat([x, padding], dim=1)
            
            # x is now (B, 512) - ready for classification head
            # Skip temporal modeling section for video models
        else:
            # ResNet: process each frame independently
            # Reshape to (B*T, 3, H, W)
            x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
            
            # Extract features: (B*T, feature_dim, 1, 1)
            x = self.features(x)
            
            # Reshape back to (B, T, feature_dim)
            x = x.view(batch_size, num_frames, -1)
            
            # Temporal modeling (only for non-video models)
            if self.use_temporal and self.lstm is not None:
                # LSTM processes temporal sequence
                x, (h_n, c_n) = self.lstm(x)  # x: (B, T, 512)
                # Use last hidden state
                x = h_n[-1]  # (B, 512) - last layer's hidden state
            else:
                # Fallback: average over time
                x = x.mean(dim=1)  # (B, feature_dim)
        
        # Classification head (reduced size)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
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
