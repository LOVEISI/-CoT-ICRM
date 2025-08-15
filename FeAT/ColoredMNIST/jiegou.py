# 导入模型类（假设它们定义在models.py中）
from models import MLP, TopMLP
import torch


# 加载权重字典
checkpoint = torch.load('G:/feta/FeAT/ColoredMNIST/feat_pretrained_model/mlp0.pth')

# 查看checkpoint中的内容
print(checkpoint.keys())


# 加载权重字典
checkpoint1 = torch.load('G:/feta/FeAT/ColoredMNIST/feat_pretrained_model/topmlp0.pth')
print(checkpoint1.keys())



# 加载模型权重