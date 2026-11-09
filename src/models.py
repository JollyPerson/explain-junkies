import torch.nn as nn
import timm


class SmallCNN(nn.Module):
    """A ~60k-parameter CNN for 28x28 greyscale images.

    Gets ~98% on MNIST after 3 CPU epochs, so the whole pipeline
    (pretrain -> junk fine-tune -> ReX) runs on a laptop.
    """

    def __init__(self, num_classes=11, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14 -> 7
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, num_classes, in_channels=3):
    """Create an untrained model by name: 'small_cnn' or any timm model."""
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, in_channels=in_channels)
    return timm.create_model(name, pretrained=False, num_classes=num_classes)
