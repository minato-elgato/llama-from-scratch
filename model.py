import torch
import torch.nn as nn

from config import ModelArgs
from rmsnorm import RMSNorm
from block import TransformerBlock

class  Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        #Token Embeddings
        self.token_emb = nn.Embedding(args.vocab_size, args.dim)

        #Stacking Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])

        #Final Layer Normazlization with RMSNorm
        self.norm = RMSNorm(args.dim, args.norm_eps)

        #Output projection -> logits over vocab
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens, start_pos):
        B,T = tokens.shape
        assert B <= self.args.max_batch_size, "Batch size should less than max_batch_size"
        assert start_pos+T <= self.args.max_seq_len, "Sequence shouldn't exceeds max_seq_len"

        # Embed tokens -> (B, T, dim)
        x = self.token_emb(tokens)

        # Pass through each Transformer block
        for layer in self.layers:
            x= layer(x, start_pos)
        
        #Final norm
        x = self.norm(x)

        # Project to vocab logits -> (B, T, vocab_size)
        logits = self.output(x)

        return logits