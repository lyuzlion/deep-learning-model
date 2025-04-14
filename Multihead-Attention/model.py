import torch
import torch.nn as nn

class Multihead_Attention(nn.Module):
    """
    multihead_attention
    根据<https://www.github.com/kyubyong/transformer>修改
    1.split+cat
    2.matmul(q,k)
    3.mask k
    4.softmax
    5.mask q
    6.matmul(attn,v)
    7.split+cat
    8.res q
    9.norm
    """

    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 C_v=None,
                 num_heads=1,
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        C_v = C_v if C_v else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.linear_V = nn.Linear(C_v, hidden_dim)
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, C_q]
        :param K: A 3d tensor with shape of [N, T_k, C_k]
        :param V: A 3d tensor with shape of [N, T_v, C_v]
        :return:
        """
        num_heads = self.num_heads
        N = Q.size()[0]

        # Linear projections
        Q_l = nn.ReLU()(self.linear_Q(Q))
        K_l = nn.ReLU()(self.linear_K(K))
        V_l = nn.ReLU()(self.linear_V(V))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        V_split = V_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(V_split, dim=0)  # (h*N, T_v, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(K).sum(dim=-1))  # (N, T_k)
        key_masks = key_masks.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, Q.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(key_masks) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = nn.Softmax(dim=2)(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(Q).sum(dim=-1))  # (N, T_q)
        query_masks = query_masks.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, K.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks  # broadcasting. (h*N, T_q, T_k)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = outputs.split(N, dim=0)  # (N, T_q, C)
        outputs = torch.cat(outputs, dim=2)

        # Residual connection
        outputs = outputs + Q_l

        # Normalize
        outputs = self.norm(outputs)  # (N, T_q, C)

        return outputs
