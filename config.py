from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096              # Hidden dimension
    n_layers: int = 32           # Number of Transformer blocks
    n_heads: int = 32            # Number of query heads for attention
    n_kv_heads: Optional[int] = None # For Grouped Query Attention (Llama 2/3)
    vocab_size: int = 32000      # Vocabulary size (Llama 2 is 32k, Llama 3 is 128k)
    multiple_of: int = 256       # Used to calculate the SwiGLU hidden layer size
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5       # Epsilon for RMSNorm to prevent division by zero
    
    # Generation parameters
    max_batch_size: int = 32
    max_seq_len: int = 2048      # Context window
