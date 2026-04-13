import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rope import RoPE

def expand_kv(x, n_repeats):
    if n_repeats == 1:
        return x
    B,H,T,D = x.shape
    x = x[:,:,None,:,:].expand(-1,-1,n_repeats,-1,-1)
    return x.reshape(B,H*n_repeats,T,D)

class GroupedQueryAttention(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_heads, n_kv_heads, d_model):
        super().__init__()
        assert d_model % n_heads == 0, "d_model should get  divide by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads should get divided by n_kv_heads"
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.n_heads
        self.n_repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        key_cache = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim)
        value_cache = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim)

        self.register_buffer("key_cache", key_cache)
        self.register_buffer("value_cache", value_cache)

        causal_mask = torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", causal_mask)

        self.rope = RoPE(self.max_seq_len, self.head_dim)

    def forward(self, x, start_pos):
        B,T,D = x.shape
        assert B <= self.max_batch_size, "B shouldn't exceed max_batch_size"
        assert start_pos + T <= self.max_seq_len, "start_pos + seq_len shouldn't exceed max_seq_len"
        assert D == self.d_model, "D should be equals to d_model"

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        #Convert x to Shape B,T,H,D
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        #Process Query and Key into RoPE
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        #Convert shape B,T,H,D to B,H,T,D
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        #Filling up the key and value cache
        self.key_cache[:B,:,start_pos:start_pos+T,:] = k
        self.value_cache[:B,:,start_pos:start_pos+T,:] = v
        
        #updating key and value
        key_history = self.key_cache[:B, :, 0:start_pos+T, :]
        value_history = self.value_cache[:B, :, 0:start_pos+T, :]

        #expanding key and values from n_kv_heads to n_heads
        key_history = expand_kv(key_history, self.n_repeats)
        value_history = expand_kv(value_history, self.n_repeats)

        #matmul query and k
        attn_weights = torch.matmul(q, key_history.transpose(-1,-2))/math.sqrt(self.head_dim)

        #causal masking
        causal_mask = self.mask[start_pos:start_pos+T,0:start_pos+T]
        attn_weights = attn_weights.masked_fill(causal_mask, -torch.inf)

        #applying softmax on attn_weights for probabilities
        attn_scores = F.softmax(attn_weights, dim=-1)

        #matmul attn_scores and value
        attn_weights = torch.matmul(attn_scores, value_history)

        #transposing from  B,H,T,D to B,T,H,D
        attn_weights = attn_weights.transpose(1,2)

        #changing Shape from B,T,H,D to B,T,D
        attn_weights = attn_weights.contiguous().view(B, T, self.d_model)

        output = self.out_proj(attn_weights)

        return output