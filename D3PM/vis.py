import sys
sys.path.append(".")

import netron
import torch.onnx
from torch.autograd import Variable
from model import D3PM, DummyX0Model

N = 2
myNet = D3PM(DummyX0Model(1, N), 1000, num_classes=N, hybrid_loss_coeff=0.0)
x = torch.rand(16, 1, 32, 32)
x = (x * (N - 1)).round().long().clamp(0, N - 1)
y = torch.randint(low=0, high=10, size=(16,))
print(y)
print(myNet(x, y))
modelData = "./demo.pth"
torch.onnx.export(myNet, (x, y), modelData)
netron.start(modelData)

