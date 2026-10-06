import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvFeatureDenoisingBlock(nn.Module):
    """Explicit deconvolutional reconstruction branch used as the proposed module."""
    def __init__(self, channels=128):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(channels, 64, 4, stride=2, padding=1)
        self.refine1 = ConvBlock(64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.refine2 = ConvBlock(32, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.refine3 = ConvBlock(16, 16)
        self.up4 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)
        self.out = nn.Sigmoid()

    def forward(self, z):
        x = torch.relu(self.up1(z))
        x = self.refine1(x)
        x = torch.relu(self.up2(x))
        x = self.refine2(x)
        x = torch.relu(self.up3(x))
        x = self.refine3(x)
        x = self.out(self.up4(x))
        return x


class SonarDeconvCNN(nn.Module):
    """PRJ-44 proposed model: CNN encoder + deconvolutional feature de-noising + classifier."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.enc1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2)

        self.deconv_denoiser = DeconvFeatureDenoisingBlock(128)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def encode(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        z = self.pool4(self.enc4(x))
        return z

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.deconv_denoiser(z)
        logits = self.classifier(z)
        return logits, reconstructed, z


class BaselineCNN(nn.Module):
    """Baseline without the proposed deconvolutional reconstruction branch."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16), nn.MaxPool2d(2),
            ConvBlock(16, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        z = self.features(x)
        return self.classifier(z)
