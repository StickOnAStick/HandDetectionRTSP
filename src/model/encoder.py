import torch



class Encoder(torch.nn.Module):

    def __init__(self, input_dim: int = 128, embed_dim: int = 256):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim)
        )