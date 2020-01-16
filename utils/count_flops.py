from models.Box_reg import Box_reg
from models.res18 import Resnet18
import torch
from thop import profile
# model=Resnet18()
model=Box_reg()
input = torch.randn(1, 1, 64, 64)
flops,params=profile(model,inputs=(input,))
print(flops,params)