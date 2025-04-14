import torch
from model import Multihead_Attention

features = torch.arange(0, 24)
features = torch.where(features < 20, features, torch.zeros_like(features))
features = features.view([2, 3, 4]).float()

attention3 = Multihead_Attention(hidden_dim=features.size()[-1])
print(attention3(features, features, features))
