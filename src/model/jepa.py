import torch
import torchvision




class Jepa(torch.nn.Module):

    def __init__(
            self, 
            input_dim: int,
            embed_dim: int,
            ema_decay: float = 0.995
        ):
        super(Jepa, self).__init__() # init module

        self.ema_decay = ema_decay

        # 1. Context Encoder (Trained via gradients)
        self.context_encoder = SimpleEncoder(input_dim, embed_dim)

        # 2. Predictor network (Trained via gradients)
        self.predictor = SimplePredictor(embed_dim)

        # 3. Target Encoder (Trained via EMA momentum tracking - No gradients!)
        self.target_encoder = copy.deepcopy(self.context_encoder)