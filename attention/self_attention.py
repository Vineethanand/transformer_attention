import torch.nn as nn
import os

class SelfAttention(nn.Module):
    """
    Self Attention class
    """
    def __init__(embed_dim=3,dk_dq_out=2, dv_out=4):
        super.__init__()
        self.dk_dq_out = dk_dq_out
        self.Wq = torch.nn.Parameter(torch.randn(embed_dim,dk_dq_out))
        self.Wk = torch.nn.Parameter(torch.randn(embed_dim,dk_dq_out))
        self.Wv = torch.nn.Parameter(torch.randn(embed_dim, dv_out))

    
    def forward(self, x):
        queries = x @ self.Wq
        keys = x @ self.Wk
        values = x @ self.Wv
    
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores/self.dk_dq_out** 0.5, dim=0)

        context_vec = attn_weights @ values
        return context_vec
