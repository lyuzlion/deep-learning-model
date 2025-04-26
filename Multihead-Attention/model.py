import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.wq = nn.Linear(hid_dim, hid_dim)
        self.wk = nn.Linear(hid_dim, hid_dim)
        self.wv = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        head_dim = self.hid_dim // self.n_heads
        
        # Project inputs
        Q, K, V = self.wq(query), self.wk(key), self.wv(value)
        
        # Reshape to [bsz, n_heads, seq_len, head_dim]
        Q = Q.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)
        K = K.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        x = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        # Combine heads and project
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.hid_dim)
        return self.fc(x)
    
query = torch.rand(64, 12, 300)
key = torch.rand(64, 10, 300)
value = torch.rand(64, 10, 300)
attention = MultiHeadAttention(hid_dim=300, n_heads=6, dropout=0.1)
output = attention(query, key, value)
print(output)