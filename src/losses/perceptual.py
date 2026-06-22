"""VGG-19 perceptual loss.

Compares features extracted from pretrained VGG-19 layers
relu_1_2, relu_2_2, relu_3_3 between prediction and target.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """Feature-space L1 loss on frozen pretrained VGG-19 activations.

    Inputs are expected in [-1, 1] and are internally remapped to the
    ImageNet-normalized range expected by VGG.
    """

    LAYERS = ("relu_1_2", "relu_2_2", "relu_3_3")
    # Indices in vgg19.features right after each target ReLU (VGG19 has 37 layers)
    _SLICE_ENDS = (4, 9, 18)

    def __init__(self, layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0)):
        super().__init__()
        if len(layer_weights) != len(self.LAYERS):
            raise ValueError(f"need {len(self.LAYERS)} layer weights")
        self.layer_weights = layer_weights

        features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        starts = (0, *self._SLICE_ENDS[:-1]) # (0, 4, 9)
        self.slices = nn.ModuleList([
            # * because Sequential wants separated arguments, not a list
            nn.Sequential(*[features[i] for i in range(start, end)])
            for start, end in zip(starts, self._SLICE_ENDS) #(0, 4, 9) (4, 9, 18)
        ])
        for param in self.parameters():
            param.requires_grad_(False) # VGG19 is frozen
        self.eval()

        # Register ImageNet constants as PyTorch buffers.
        # Shape (1, 3, 1, 1) enables broadcasting across [B, 3, H, W] batches.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Remap inputs from dataset range [-1, 1] to ImageNet standard [0, 1] 
        and then apply channel-wise mean/std normalization for VGG."""
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x, y = self._normalize(prediction), self._normalize(target)
        loss = torch.zeros((), device=prediction.device)
        for weight, vgg_slice in zip(self.layer_weights, self.slices):
            x, y = vgg_slice(x), vgg_slice(y)
            loss = loss + weight * self.criterion(x, y)
        return loss
