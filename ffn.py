import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int | None = None, multiple_of: int = 256, ffn_dim_multiplier=None)->None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int((8 * d_in) / 3)
        
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)

        self.w1 = nn.Linear(d_in, hidden_dim, bias=False)
        self.v = nn.Linear(d_in, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_in, bias=False)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        swish_gate = F.silu(self.w1(x))
        value = self.v(x)
        gated = swish_gate * value
        output = self.w2(gated)

        return output