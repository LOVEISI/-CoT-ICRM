# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # CIFAR10 数据集加载
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# import torch.nn.functional as F
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from torchvision import models
# # Modified ResNet-18
# class ModifiedResNet18(nn.Module):
#     def __init__(self):
#         super(ModifiedResNet18, self).__init__()
#         # 加载预训练的 ResNet-18 模型
#         self.resnet18 = models.resnet18(pretrained=True)
        
#         # 修改第一层卷积层，让其输出通道数为 3 而不是 64
#         self.resnet18.conv1 = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3, bias=False)

#         # 移除最后的分类层
#         self.resnet18.fc = nn.Identity()

#     def forward(self, x):
#         return self.resnet18(x)
# # 使用GPT2的解码器架构
# class ICRMTransformer(nn.Module):
#     def __init__(self):
#         super(ICRMTransformer, self).__init__()
#         # 加载预训练GPT-2模型
#         self.model = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.model.resize_token_embeddings(256)  # 确保输入层的维度
#          # 使用一个预训练的CNN模型（例如ResNet）来提取图像特征
#         # self.cnn = models.resnet18(pretrained=True)  # 使用ResNet-18
#         # self.cnn.fc = nn.Identity()  # 移除最后的分类层
#         # 使用修改后的 ResNet-18 来提取图像特征
#         self.cnn = ModifiedResNet18()  # 使用修改后的ResNet-18

#     def forward(self, x, context):
#         if x.size(1) != 3:
#             raise ValueError(f"Expected 3 channels in input, but got {x.size(1)} channels.")

#          # 将上下文和目标图像展平为一维向量
#         # context = context.view(context.size(0), -1)  # 展开为二维 [batch_size, channels * height * width]
#         # x = x.view(x.size(0), -1)  # 展开为二维 [batch_size, channels * height * width]
#         x_features = self.cnn(x)  # 获取图像特征，形状：[batch_size, feature_dim]
#         context = x_features
#         # 上下文拼接到输入中
#         # input_ids = torch.cat([context, x], dim=1)  # 拼接当前样本和上下文
#         input_ids = torch.cat([context, x_features], dim=1)  # 拼接特征
#         input_ids = input_ids.long()
#         outputs = self.model(input_ids)
#         return outputs.logits

# # 定义损失函数
# criterion = nn.CrossEntropyLoss()

# # 初始化模型
# model = ICRMTransformer()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # 训练过程
# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()

#         # 从批次中获取上下文数据（未标记的样本，假设这里从训练集中获取）
#         context = data[:, :-1, :]  # 假设 context 是先前的图像（上下文不包括最后一个像素）

#         # 当前样本
#         x = data[:, -1, :]

#         # 前向传播
#         output = model(x, context)
#         loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader)}")

# # 测试阶段
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for data, target in test_loader:
#         context = data[:, :-1, :]  # 获取上下文
#         x = data[:, -1, :]

#         # 获取预测
#         output = model(x, context)
#         _, predicted = torch.max(output, 1)

#         total += target.size(0)
#         correct += (predicted == target).sum().item()

# accuracy = 100 * correct / total
# print(f'Test Accuracy: {accuracy}%')


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from torchvision import models

# CIFAR10 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Modified ResNet-18
class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        # 加载预训练的 ResNet-18 模型
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用最新的预训练权重
        
        # 修改第一层卷积层，让其输出通道数为 3 而不是 64
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 保持为 64 通道
        
        # 修改 Batch Normalization 层的 num_features 为 64
        self.resnet18.bn1 = nn.BatchNorm2d(64)

        # 移除最后的分类层
        self.resnet18.fc = nn.Identity()

    def forward(self, x):
        return self.resnet18(x)

# 使用GPT2的解码器架构
class ICRMTransformer(nn.Module):
    def __init__(self):
        super(ICRMTransformer, self).__init__()
        # 加载预训练GPT-2模型
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(256)  # 确保输入层的维度
        
        # 使用修改后的 ResNet-18 来提取图像特征
        self.cnn = ModifiedResNet18()  # 使用修改后的ResNet-18

    def forward(self, x):
        # 提取图像特征
        x_features = self.cnn(x)  # 获取图像特征，形状：[batch_size, feature_dim]
        
        # 直接使用图像特征作为上下文
        context = x_features

        # 将上下文与目标样本拼接（注意：GPT-2 期望的是整数索引类型）
        input_ids = torch.cat([context, x_features], dim=1)  # 拼接特征
        input_ids = input_ids.long()  # 将其转换为整数类型
        
        # 输入给GPT-2模型
        outputs = self.model(input_ids)
        return outputs.logits

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 初始化模型
model = ICRMTransformer()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练过程
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # 前向传播
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader)}")

# 测试阶段
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy}%')
