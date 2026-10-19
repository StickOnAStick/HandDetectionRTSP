import torch
import copy
from collections.abc import Sequence
from typing import Optional

from .encoder import Encoder
from .predictor import Predictor

class Jepa(torch.nn.Module):

    def __init__(
            self, 
            input_dim: int = 1024,
            embed_dim: int = 512,
            latent_dim: int = 512,
            encoder_hidden_dim: Optional[Sequence[int]] = (768,),
            predictor_hidden_dim: Optional[Sequence[int]] = (768, 768),
            ema_decay: float = 0.995
        ):
        super().__init__()

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")

        self.ema_decay = ema_decay

        # 1. Context Encoder (Trained via gradients)
        self.context_encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            embed_dim=embed_dim,
        )

        # 2. Predictor network (Trained via gradients)
        self.predictor = Predictor(
            input_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=embed_dim,
        )

        # 3. Target Encoder (Trained via EMA momentum tracking - No gradients!)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

        self.loss_fn = torch.nn.SmoothL1Loss()

    def forward(self, context_input, latent):
        """Predict a target embedding from a context vector and conditioning latent."""
        context_embedding = self.context_encoder(context_input)
        return self.predictor(context_embedding, latent)

    @torch.no_grad()
    def encode_target(self, target_input):
        """Create a stable target embedding without adding it to the gradient graph."""
        return self.target_encoder(target_input)

    def compute_loss(self, context_input, target_input, latent):
        """Return the JEPA Smooth L1 loss and useful detached outputs for logging."""
        prediction = self(context_input, latent)
        target = self.encode_target(target_input)
        loss = self.loss_fn(prediction, target)
        return loss, prediction.detach(), target

    @torch.no_grad()
    def update_target_encoder(self):
        """Move target weights slowly toward the gradient-trained context encoder."""
        for target_param, context_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target_param.mul_(self.ema_decay).add_(
                context_param, alpha=1.0 - self.ema_decay
            )

        # This matters for encoders with non-parameter state, e.g. BatchNorm.
        for target_buffer, context_buffer in zip(
            self.target_encoder.buffers(), self.context_encoder.buffers()
        ):
            target_buffer.copy_(context_buffer)
