import torch
import torch.nn as nn

class RoPE(nn.Module):
  def __init__(self, max_seq_len: int = 4096, head_dim: int = 128, base: float=10000.0)->None:
    super().__init__()
    index = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta = torch.exp(index * -torch.log(torch.tensor(base))/head_dim)

    pos = torch.arange(0, max_seq_len, dtype=torch.float32)
    angles = torch.outer(pos, theta)

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    self.register_buffer("sin_cached", sin)
    self.register_buffer("cos_cached", cos)

  def forward(self, x: torch.Tensor)->torch.Tensor:
    seq_len = x.shape[1]

    sin = self.sin_cached[:seq_len, :]
    cos = self.cos_cached[:seq_len, :]
    
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = x_even_rot
    x_rot[..., 1::2] = x_odd_rot

    return x_rot