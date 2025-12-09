# SlowOnly model architecture for HRI30

import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, BACKBONE, PRETRAINED_DATASET


class SlowOnlyModel(nn.Module):
    def __init__(self, num_classes=30, dropout_rate=0.6, **kwargs):
        super(SlowOnlyModel, self).__init__()
        
        # 1. Load the full pre-trained model
        full_model = models.video.r2plus1d_18(pretrained=True)
        
        # 2. CRITICAL FIX: Remove the last 'fc' layer
        # list(children())[:-1] keeps everything EXCEPT the last layer (fc)
        # This includes the 3D convolutions and the final Average Pooling
        self.backbone = nn.Sequential(*list(full_model.children())[:-1])
        
        # 3. Define your new classification head
        # The output of r2plus1d_18 backbone is 512 dimensions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape from DataLoader: (Batch, Time, Channels, Height, Width)
        # e.g., (4, 16, 3, 224, 224)

        # 4. FIX INPUT SHAPE: Permute to (Batch, Channels, Time, Height, Width)
        # PyTorch Video models expect channels first (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 5. Extract Features
        # Input: (4, 3, 16, 224, 224) -> Output: (4, 512, 1, 1, 1)
        x = self.backbone(x)
        
        # 6. Flatten
        # Output: (4, 512)
        x = x.flatten(1)
        
        # 7. Classify
        # Output: (4, 30)
        x = self.classifier(x)
        
        return x


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
