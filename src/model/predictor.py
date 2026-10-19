import torch
from collections.abc import Sequence
from typing import Optional

class Predictor(torch.nn.Module):

    def __init__(
            self, 
            input_dim: int = 512, 
            latent_dim: int = 512,
            hidden_dim: Optional[Sequence[int]] = (768, 768),
            output_dim: int = 512,
        ):
        super().__init__()

        # None or [] creates a direct (input + latent) -> output predictor.
        hidden_dims = [] if hidden_dim is None else list(hidden_dim)
        dims = [input_dim + latent_dim, *hidden_dims, output_dim]
        layers: list[torch.nn.Module] = []

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if index < len(dims) - 2:
                layers.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, current_embeding, latent_embedding):
        x = torch.cat([current_embeding, latent_embedding], dim=1)
        return self.net(x)
