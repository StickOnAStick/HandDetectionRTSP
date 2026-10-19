import torch
from collections.abc import Sequence
from typing import Optional



class Encoder(torch.nn.Module):

    def __init__(
            self, 
            input_dim: int = 1024, 
            hidden_dim: Optional[Sequence[int]] = (768,),
            embed_dim: int = 512
        ):
        super().__init__()

        # None or [] creates a direct input_dim -> embed_dim encoder.
        hidden_dims = [] if hidden_dim is None else list(hidden_dim)
        dims = [input_dim, *hidden_dims, embed_dim]
        layers: list[torch.nn.Module] = []

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if index < len(dims) - 2:
                layers.append(torch.nn.SiLU())

        layers.append(torch.nn.LayerNorm(embed_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
