import torch
from model import MultiheadAttention

features = torch.arange(0, 24)
features = torch.where(features < 20, features, torch.zeros_like(features))
features = features.view([2, 3, 4]).float()
print(features)
attention3 = MultiheadAttention(hidden_dim=features.size()[-1])
print(attention3(features, features, features))
