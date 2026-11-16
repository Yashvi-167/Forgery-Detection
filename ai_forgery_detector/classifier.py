"""
AIForgeryClassifier
===================
Binary classifier to distinguish AI-generated / diffusion-based forgeries
(e.g. Stable Diffusion inpainting, SDXL, Adobe Firefly, FLUX) from
traditional forgeries or authentic images.

Architecture
------------
Two parallel branches are fused before the final linear head:

1. **RGB branch** — ResNet-18 backbone (last 2 layers unfrozen) produces a
   512-d spatial feature vector.

2. **Frequency branch** — the FFT magnitude spectrum of the grayscale image
   is computed and fed through a lightweight 4-layer CNN to produce a 128-d
   feature.  Diffusion models leave characteristic periodic patterns in the
   frequency domain that CNN-based detectors often miss.

Both feature vectors are concatenated (640-d) and passed through a small
MLP to output logits for 2 classes:
    0 → traditional / authentic
    1 → AI-generated forgery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ---------------------------------------------------------------------------
# Frequency-domain branch
# ---------------------------------------------------------------------------

class FrequencyBranch(nn.Module):
    """
    Converts an input image to its 2-D FFT magnitude spectrum and runs a
    lightweight CNN on it.

    Input  : (B, 3, H, W)  — values in [0, 1] after normalisation
    Output : (B, 128)
    """

    def __init__(self):
        super().__init__()
        # Grayscale conversion weights (BT.601)
        self.register_buffer(
            "gray_weight",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        )
        # Small 4-conv CNN on the log-magnitude spectrum
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # /2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # /16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Convert to grayscale: (B, 1, H, W)
        gray = (x * self.gray_weight).sum(dim=1, keepdim=True)

        # 2-D FFT magnitude spectrum (log-compressed, normalised)
        fft_out = torch.fft.fft2(gray, norm="ortho")           # complex
        magnitude = torch.abs(fft_out)                          # (B,1,H,W)
        # shift DC to centre
        magnitude = torch.fft.fftshift(magnitude)
        # log compression, avoid log(0)
        magnitude = torch.log1p(magnitude)
        # normalise per sample to [0,1]
        b = magnitude.shape[0]
        mag_min = magnitude.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        mag_max = magnitude.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

        feat = self.cnn(magnitude)          # (B, 128, h, w)
        feat = self.pool(feat)              # (B, 128, 1, 1)
        return feat.view(feat.size(0), -1)  # (B, 128)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class AIForgeryClassifier(nn.Module):
    """
    Lightweight binary classifier for AI-generated image forgery detection.

    Parameters
    ----------
    backbone : str
        One of 'resnet18' (default), 'resnet50'.
    pretrained : bool
        Whether to load ImageNet weights for the RGB backbone.
    freeze_rgb_backbone : bool
        If True, freeze all ResNet layers except layer3, layer4, and fc.
        Recommended for fine-tuning on small AI-forgery datasets.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_rgb_backbone: bool = True,
    ):
        super().__init__()

        # ----- RGB branch -----
        if backbone == "resnet18":
            try:
                # torchvision >= 0.13
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                rgb_net = models.resnet18(weights=weights)
            except AttributeError:
                # torchvision < 0.13 fallback
                rgb_net = models.resnet18(pretrained=pretrained)
            rgb_feat_dim = 512
        elif backbone == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                rgb_net = models.resnet50(weights=weights)
            except AttributeError:
                rgb_net = models.resnet50(pretrained=pretrained)
            rgb_feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove final FC; keep everything up to avgpool
        self.rgb_backbone = nn.Sequential(
            rgb_net.conv1,
            rgb_net.bn1,
            rgb_net.relu,
            rgb_net.maxpool,
            rgb_net.layer1,
            rgb_net.layer2,
            rgb_net.layer3,
            rgb_net.layer4,
            rgb_net.avgpool,  # AdaptiveAvgPool2d(1,1)
        )

        # Optionally freeze early layers
        if freeze_rgb_backbone:
            freeze_until = 6  # freeze conv1 through layer2 (indices 0-5)
            for i, child in enumerate(self.rgb_backbone.children()):
                if i < freeze_until:
                    for param in child.parameters():
                        param.requires_grad = False

        # ----- Frequency branch -----
        self.freq_branch = FrequencyBranch()

        # ----- Fusion MLP -----
        fused_dim = rgb_feat_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor  (B, 3, H, W)
            Normalised input image (SAM normalisation or ImageNet norm).

        Returns
        -------
        logits : torch.Tensor  (B, 2)
            Raw logits.  Apply softmax / argmax for prediction.
        features : torch.Tensor  (B, fused_dim)
            Intermediate fused feature (useful for contrastive losses).
        """
        # RGB features
        rgb_feat = self.rgb_backbone(x)             # (B, C, 1, 1)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # (B, C)

        # Frequency features
        freq_feat = self.freq_branch(x)             # (B, 128)

        # Fuse
        features = torch.cat([rgb_feat, freq_feat], dim=1)  # (B, C+128)
        logits = self.fusion(features)                       # (B, 2)
        return logits, features

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_detector(self, path: str):
        """Save all state-dict parameters."""
        torch.save(self.state_dict(), path)
        print(f"[AIForgeryClassifier] Saved to {path}")

    def load_detector(self, path: str, map_location: str = "cpu"):
        """Load state-dict, with strict=False for forward compatibility."""
        state_dict = torch.load(path, map_location=map_location)
        if isinstance(self, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.module.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)
        print(f"[AIForgeryClassifier] Loaded from {path}")
