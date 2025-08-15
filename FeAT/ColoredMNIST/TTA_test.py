import torch
import torch.nn as nn
import torch.optim as optim
from mydatasets import coloredmnist
from models import MLP, TopMLP

# 加载目标域数据
_, target_envs = coloredmnist(0.1, 0.2, 0.25, int_target=False)
test_x = torch.cat([env['images'] for env in target_envs])
test_y = torch.cat([env['labels'] for env in target_envs])

# 特征提取器（源域预训练）
mlp = MLP(hidden_dim=128, input_dim=2*14*14)
# mlp.load_state_dict(torch.load("G:/feta/FeAT/ColoredMNIST/results/erm_pretrain/mlp2.pth"))
mlp.load_state_dict(torch.load("G:/feta/FeAT/ColoredMNIST/feat_pretrained_model/mlp0.pth"))


topmlp = TopMLP(hidden_dim=128, n_top_layers=1, n_targets=1)
topmlp.load_state_dict(torch.load("G:/feta/FeAT/ColoredMNIST/feat_pretrained_model/topmlp0.pth"))
# topmlp.load_state_dict(torch.load("G:/feta/FeAT/ColoredMNIST/results/erm_pretrain/topmlp2.pth"))

model = nn.Sequential(mlp, topmlp)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# TTA微调
model.train()
for epoch in range(3):
    outputs = model(test_x)
    pseudo_labels = (torch.sigmoid(outputs) > 0.5).float().detach()
    loss = criterion(outputs, pseudo_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"TTA Epoch {epoch}, Loss: {loss.item():.4f}")

# 评估
model.eval()
with torch.no_grad():
    logits = model(test_x)
    pred = (torch.sigmoid(logits) > 0.5).float()
    acc = (pred == test_y).float().mean().item()
    print(f"TTA Adaptation Accuracy: {acc:.2%}")
