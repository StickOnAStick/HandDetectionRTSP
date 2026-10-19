import torch

from .jepa import Jepa


def train_step(
    model: Jepa,
    optimizer: torch.optim.Optimizer,
    context_input: torch.Tensor,
    target_input: torch.Tensor,
    latent: torch.Tensor,
) -> float:
    """Run one JEPA optimization step and return its scalar loss."""
    model.train()
    model.target_encoder.eval()

    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = model.compute_loss(context_input, target_input, latent)
    loss.backward()
    optimizer.step()
    model.update_target_encoder()

    return loss.item()
