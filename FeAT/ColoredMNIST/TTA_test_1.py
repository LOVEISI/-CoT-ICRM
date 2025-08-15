import torch
import torch.nn as nn
import torch.optim as optim
from mydatasets import coloredmnist
from models import MLP, TopMLP
import argparse
import matplotlib.pyplot as plt

# ----------------- 更新 BN running stats -----------------
def update_bn_stats(model, data):
    model.train()
    with torch.no_grad():
        _ = model(data)

# ----------------- 熵计算函数 -----------------
def entropy_loss_fn(logits):
    probs = torch.sigmoid(logits)
    return - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).mean()

# ----------------- 主函数 -----------------
def main(args):
    # 加载目标域数据
    _, target_envs = coloredmnist(0.1, 0.2, 0.25, int_target=False)
    test_x = torch.cat([env['images'] for env in target_envs])
    test_y = torch.cat([env['labels'] for env in target_envs]).float().unsqueeze(1)

    # 加载模型
    mlp = MLP(hidden_dim=128, input_dim=2 * 14 * 14)
    topmlp = TopMLP(hidden_dim=128, n_top_layers=1, n_targets=1)
    mlp.load_state_dict(torch.load(args.mlp_path))
    topmlp.load_state_dict(torch.load(args.topmlp_path))
    model = nn.Sequential(mlp, topmlp)

    # 更新BN统计
    if args.use_bn:
        print("Updating BN statistics...")
        update_bn_stats(model, test_x)

    # 设置哪些参数需要优化
    if args.use_tent:
        print("Using Tent strategy: only updating BN parameters")
        for name, param in model.named_parameters():
            param.requires_grad = ("bn" in name.lower())
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # ---------- TTA训练 ----------
    loss_history = []

    if args.use_entropy or args.use_tent:
        model.train()
        for epoch in range(args.epochs):
            logits = model(test_x)
            loss = entropy_loss_fn(logits)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"TTA Epoch {epoch}, Loss: {loss.item():.4f}")

    # ---------- 评估 ----------
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == test_y).float().mean().item()
        print(f"\n🚀 Target Domain Accuracy: {acc * 100:.2f}%")

    # ---------- 绘制训练 Loss ----------
    if loss_history:
        plt.plot(loss_history)
        plt.title("TTA Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ----------------- CLI 入口 -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp_path', type=str, default="./feat_pretrained_model/mlp0.pth")
    parser.add_argument('--topmlp_path', type=str, default="./feat_pretrained_model/topmlp0.pth")
    parser.add_argument('--use_bn', action='store_true', help='更新 BN running stats')
    parser.add_argument('--use_entropy', action='store_true', help='是否使用熵最小化')
    parser.add_argument('--use_tent', action='store_true', help='是否仅更新BN参数（Tent）')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    main(args)
