# SlowOnly model architecture for HRI30

import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    BACKBONE,
    FREEZE_BACKBONE,
    HEAD_DROPOUT,
    MODEL_TYPE,
    NUM_CLASSES,
    PRETRAINED_DATASET,
)


def _resolve_resnet_weights(backbone_name, pretrained_source):
    """
    Map backbone to torchvision weight enum (ImageNet) when available.
    Falls back to None if weights cannot be resolved (e.g., offline).
    """
    if pretrained_source != "imagenet":
        return None

    weight_map = {
        "resnet18": getattr(models, "ResNet18_Weights", None),
        "resnet34": getattr(models, "ResNet34_Weights", None),
        "resnet50": getattr(models, "ResNet50_Weights", None),
        "resnet101": getattr(models, "ResNet101_Weights", None),
    }
    weight_enum = weight_map.get(backbone_name)
    if weight_enum is None:
        return None

    # Prefer V2 weights when available, otherwise V1
    return getattr(weight_enum, "IMAGENET1K_V2", None) or getattr(
        weight_enum, "IMAGENET1K_V1", None
    )


def build_backbone(backbone_name=BACKBONE, pretrained_source=PRETRAINED_DATASET):
    """Create a 2D CNN backbone and return the feature extractor and feature dim."""
    weights = _resolve_resnet_weights(backbone_name, pretrained_source)
    backbone_fn = getattr(models, backbone_name)

    try:
        backbone_model = backbone_fn(weights=weights)
    except Exception as err:
        print(f"⚠ Pre-trained weights unavailable ({err}). Using random init.")
        backbone_model = backbone_fn(weights=None)

    feature_dim = backbone_model.fc.in_features
    feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
    return feature_extractor, feature_dim


class SlowOnlyModel(nn.Module):
    """
    SlowOnly-style architecture using a 2D CNN backbone + temporal average pooling.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        pretrained=PRETRAINED_DATASET,
        dropout_rate=HEAD_DROPOUT,
        freeze_backbone=FREEZE_BACKBONE,
    ):
        super(SlowOnlyModel, self).__init__()

        self.feature_extractor, self.feature_dim = build_backbone(
            backbone, pretrained
        )

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Classification head with dropout regularisation
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        print("✓ SlowOnly Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  Pre-trained: {pretrained if pretrained else 'No'}")
        print(f"  Feature Dimension: {self.feature_dim}")
        print(f"  Freeze backbone: {freeze_backbone}")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, 3, H, W)
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        batch_size, num_frames = x.shape[0], x.shape[1]

        # Merge batch and temporal dimensions to reuse 2D CNN
        x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
        x = self.feature_extractor(x)  # (B*T, feature_dim, 1, 1)
        x = x.view(batch_size, num_frames, -1)

        # Temporal average pooling
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


class CNNAvgPoolModel(nn.Module):
    """
    Baseline: 2D CNN backbone with temporal average pooling and a single FC head.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        pretrained=PRETRAINED_DATASET,
        dropout_rate=HEAD_DROPOUT,
        freeze_backbone=FREEZE_BACKBONE,
    ):
        super(CNNAvgPoolModel, self).__init__()
        self.feature_extractor, self.feature_dim = build_backbone(backbone, pretrained)

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

        print("✓ CNN+AvgPool Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  Pre-trained: {pretrained if pretrained else 'No'}")
        print(f"  Freeze backbone: {freeze_backbone}")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, 3, H, W)
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        batch_size, num_frames = x.shape[0], x.shape[1]

        x = x.view(batch_size * num_frames, x.shape[2], x.shape[3], x.shape[4])
        x = self.feature_extractor(x)
        x = x.view(batch_size, num_frames, -1)

        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


class CNNLSTMModel(nn.Module):
    """
    CNN backbone feeding an LSTM for temporal modelling.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        pretrained=PRETRAINED_DATASET,
        lstm_hidden=512,
        lstm_layers=2,
        dropout_rate=HEAD_DROPOUT,
        freeze_backbone=FREEZE_BACKBONE,
    ):
        super(CNNLSTMModel, self).__init__()

        self.cnn, self.cnn_feature_dim = build_backbone(backbone, pretrained)
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # LSTM for temporal dynamics
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_hidden, num_classes),
        )

        print("✓ CNN-LSTM Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  LSTM Hidden: {lstm_hidden}")
        print(f"  Freeze backbone: {freeze_backbone}")

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
        x, _ = self.lstm(x)

        # Use last hidden state
        x = x[:, -1, :]

        logits = self.head(x)
        return logits


def build_model(
    model_type=MODEL_TYPE,
    backbone=BACKBONE,
    pretrained=PRETRAINED_DATASET,
    dropout_rate=HEAD_DROPOUT,
    freeze_backbone=FREEZE_BACKBONE,
):
    """Factory to build a model by name."""
    model_type = model_type.lower()
    if model_type == "slowonly":
        return SlowOnlyModel(
            num_classes=NUM_CLASSES,
            backbone=backbone,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )
    if model_type == "cnn_avgpool":
        return CNNAvgPoolModel(
            num_classes=NUM_CLASSES,
            backbone=backbone,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )
    if model_type == "cnn_lstm":
        return CNNLSTMModel(
            num_classes=NUM_CLASSES,
            backbone=backbone,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unknown architecture: {model_type}")


def load_model(model_path, device, architecture="slowonly"):
    """Load a saved model"""
    model = build_model(model_type=architecture)
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
