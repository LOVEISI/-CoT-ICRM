import torch
import torch.nn as nn
import torch.optim as optim
from mydatasets import coloredmnist
from models import MLP, TopMLP

# 加载数据
source_envs, _ = coloredmnist(0.25, 0.1, 0.2, int_target=False)
_, target_envs = coloredmnist(0.1, 0.2, 0.25, int_target=False)

source_x = torch.cat([env['images'] for env in source_envs])
source_y = torch.cat([env['labels'] for env in source_envs])

target_x = torch.cat([env['images'] for env in target_envs])

# 特征提取器（源域预训练）
mlp = MLP(hidden_dim=128, input_dim=2*14*14)
mlp.load_state_dict(torch.load("G:/feta/FeAT/ColoredMNIST/feat_pretrained_model/mlp0.pth"))
mlp.eval()

# 域分类器
domain_classifier = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 2), nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(list(mlp.parameters()) + list(domain_classifier.parameters()), lr=1e-4)

# DA 训练过程
epochs = 5
for epoch in range(epochs):
    mlp.train()
    domain_classifier.train()

    inputs = torch.cat([source_x, target_x])
    domain_labels = torch.cat([torch.zeros(len(source_x)), torch.ones(len(target_x))]).long()

    features = mlp(inputs)
    preds = domain_classifier(features)
    loss = criterion(preds, domain_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print("DA Adaptation finished.")

# ========== 任务分类器训练和测试代码（追加） ==========
# 任务分类器：用来分类任务标签 (0/1)，而非域标签
task_classifier = TopMLP(hidden_dim=128, n_top_layers=1, n_targets=1)
task_classifier = task_classifier.eval()

# 训练任务分类器时，不更新 mlp（特征提取器）参数
optimizer_task = optim.Adam(task_classifier.parameters(), lr=1e-3)
loss_fn_task = nn.BCEWithLogitsLoss()

# 训练任务分类器（只用源域数据）
source_y_float = source_y.float().view(-1, 1)  # (N, 1)


task_epochs = 10
for epoch in range(task_epochs):
    task_classifier.train()
    mlp.eval()  # 不更新特征提取器

    with torch.no_grad():
        source_features = mlp(source_x)

    logits = task_classifier(source_features)
    loss = loss_fn_task(logits, source_y_float)

    optimizer_task.zero_grad()
    loss.backward()
    optimizer_task.step()

    print(f'Task Epoch {epoch}, Task Loss: {loss.item():.4f}')

# 测试任务分类器在目标域上的表现
task_classifier.eval()
mlp.eval()
with torch.no_grad():
    target_features = mlp(target_x)
    target_logits = task_classifier(target_features)
    target_preds = (torch.sigmoid(target_logits) > 0.5).long()

    target_labels = torch.cat([env['labels'] for env in target_envs])

    # 确保preds和labels维度一致
    correct = (target_preds.view(-1) == target_labels.view(-1)).sum().item()
    acc = correct / len(target_labels)

print(f"\n✅ Target domain accuracy after DA: {acc * 100:.2f}%")


