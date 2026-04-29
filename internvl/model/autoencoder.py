import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        # 编码器：单层线性 + ReLU
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # 解码器：单层线性 + Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)           # 压缩表示
        decoded = self.decoder(encoded)     # 重建输入
        return decoded

    def compute_reconstruction_loss(self, embeddings):
        self.eval()
        with torch.no_grad():   # 不计算梯度
            reconstruction = self(embeddings)
            # 逐样本的MSE损失
            losses = torch.mean((reconstruction - embeddings) ** 2, dim=1)
        return losses
