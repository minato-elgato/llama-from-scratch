import torch
import torch.nn as nn

from config import ModelArgs
from rmsnorm import RMSNorm
from attention import GroupedQueryAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        #Normalization Layer for Attention
        self.norm_attention = RMSNorm(args.dim, args.norm_eps)

        #Attention Block
        self.attention = GroupedQueryAttention(
            args.max_batch_size,
            args.max_seq_len,
            args.n_heads,
            args.n_kv_heads,
            args.dim
                                               )
        
        #Normalization Layer for Feed Forward
        self.norm_ffn = RMSNorm(args.dim, args.norm_eps)

        #Feed Forward Block
        self.feed_forward = FeedForward(args.dim, ffn_dim_multiplier=args.ffn_dim_multiplier , multiple_of=args.multiple_of)

    def forward(self, x, start_pos):
        x = x + self.attention(self.norm_attention(x), start_pos)
        output = x + self.feed_forward(self.norm_ffn(x))

        return output