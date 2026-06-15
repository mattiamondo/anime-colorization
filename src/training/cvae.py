"""Trainer for the Conditional VAE (variant 5)."""

from .base import BaseTrainer


class CVAETrainer(BaseTrainer):
    """CVAE with reconstruction + KL divergence loss.

    Args:
        beta: weight of the KL term (consider KL annealing to avoid
            posterior collapse).
    """

    def __init__(self, *, beta: float = 1.0, **kwargs):
        # TODO: build ConditionalVAE, optimizer, then call super().__init__
        raise NotImplementedError

    def training_step(self, batch):
        # TODO: reconstruction loss + beta * KL
        raise NotImplementedError

    def validation_step(self, batch):
        # TODO: val ELBO / reconstruction for checkpoint selection
        raise NotImplementedError
