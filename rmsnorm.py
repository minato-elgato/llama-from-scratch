import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5)->None:
        super().__init__()

        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = torch.pow(x, 2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.epsilon)
        return x_norm

    def forward(self, x: torch.Tensor)->torch.Tensor:
        outputs = self._norm(x.float()).to(x.dtype)
        return outputs * self.weight