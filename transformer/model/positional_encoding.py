import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        position = torch.arange(0, max_len, device=device).unsqueeze(dim=1).float() # 在指定维度dim=1处增加一个新的维度

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))

    def forward(self, x):
        bs, seq_len = x.size()
        return self.encoding[:seq_len, :]