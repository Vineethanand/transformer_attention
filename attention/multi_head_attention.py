import torch.nn as nn
from .self_attention import SelfAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,embed_dim=6, dk_dq_out=2, dv_out=4, num_heads = 3):
        super.__init__()
        self.heads = nn.ModuleList([SelfAttention(embed_dim=embed_dim,
                                                 dk_dq_out=dk_dq_out,
                                                 dv_out=dv_out)) for _ in num_heads]
    

    def forward(self, x):
        multi_head_context_vec = torch.cat([head(x) for head in self.heads], dim=-1)

        return multi_head_context_vec