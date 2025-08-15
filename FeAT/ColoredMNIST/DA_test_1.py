import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mydatasets import coloredmnist
from models import MLP, TopMLP
from torch.autograd import Function

# ------------------ Gradient Reversal Layer ------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

# ------------------ Load Data ------------------
source_envs, _ = coloredmnist(0.25, 0.1, 0.2, int_target=False) 	#训练集（train）中颜色与标签的相关性，验证集（val）中颜色错误的概率，测试集中颜色错误的概率
_, target_envs = coloredmnist(0.1, 0.2, 0.25, int_target=False)

source_x = torch.cat([env['images'] for env in source_envs])
source_y = torch.cat([env['labels'] for env in source_envs]).float().unsqueeze(1)

target_x = torch.cat([env['images'] for env in target_envs])
target_y = torch.cat([env['labels'] for env in target_envs]).float().unsqueeze(1)

# ------------------ Load Pretrained Feature Extractor ------------------
mlp = MLP(hidden_dim=128, input_dim=2*14*14)
mlp.load_state_dict(torch.load("G:/feta/FeAT/ColoredMNIST/results/erm_pretrain/mlp0.pth"))
mlp.train()

# ------------------ Define Classifiers ------------------
task_classifier = TopMLP(hidden_dim=128, n_top_layers=1, n_targets=1)
domain_classifier = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1), nn.Sigmoid()
)

# ------------------ Loss, Optimizer ------------------
criterion_task = nn.BCEWithLogitsLoss()
criterion_domain = nn.BCELoss()

optimizer = optim.Adam(
    list(mlp.parameters()) +
    list(task_classifier.parameters()) +
    list(domain_classifier.parameters()),
    lr=1e-4
)

# ------------------ Train DANN ------------------
epochs = 1000
batch_size = 512
alpha = 0.5
num_batches = min(len(source_x), len(target_x)) // batch_size

# 🔁 Add logging
task_losses = []
domain_losses = []
accuracies = []

for epoch in range(epochs):
    perm_source = torch.randperm(len(source_x))
    perm_target = torch.randperm(len(target_x))

    total_task_loss, total_domain_loss = 0.0, 0.0

    for i in range(num_batches):
        optimizer.zero_grad()

        source_idx = perm_source[i*batch_size:(i+1)*batch_size]
        target_idx = perm_target[i*batch_size:(i+1)*batch_size]

        source_batch_x = source_x[source_idx]
        source_batch_y = source_y[source_idx].squeeze(-1)
        target_batch_x = target_x[target_idx]

        source_feat = mlp(source_batch_x)
        target_feat = mlp(target_batch_x)

        task_logits = task_classifier(source_feat)
        loss_task = criterion_task(task_logits, source_batch_y)

        domain_feat = torch.cat([source_feat, target_feat])
        domain_feat_grl = grad_reverse(domain_feat, alpha)

        domain_logits = domain_classifier(domain_feat_grl).view(-1)
        domain_labels = torch.cat([
            torch.zeros(batch_size),
            torch.ones(batch_size)
        ]).to(domain_logits.device)

        loss_domain = criterion_domain(domain_logits, domain_labels)

        loss = loss_task + loss_domain
        loss.backward()
        optimizer.step()

        total_task_loss += loss_task.item()
        total_domain_loss += loss_domain.item()

    avg_task_loss = total_task_loss / num_batches
    avg_domain_loss = total_domain_loss / num_batches
    task_losses.append(avg_task_loss)
    domain_losses.append(avg_domain_loss)

    # 🎯 Evaluate on target
    mlp.eval()
    task_classifier.eval()
    with torch.no_grad():
        target_feat = mlp(target_x)
        task_logits = task_classifier(target_feat)
        preds = (torch.sigmoid(task_logits) > 0.5).float()
        correct = (preds.squeeze() == target_y.squeeze()).float().sum()
        acc = correct / target_y.size(0)
        accuracies.append(acc.item())
    mlp.train()
    task_classifier.train()

    print(f"Epoch [{epoch+1}/{epochs}] Task Loss: {avg_task_loss:.4f}, Domain Loss: {avg_domain_loss:.4f}, Target Acc: {acc.item()*100:.2f}%")

# ------------------ Plot ------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(task_losses, label="Task Loss")
plt.plot(domain_losses, label="Domain Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in accuracies], label="Target Domain Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Target Domain Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
