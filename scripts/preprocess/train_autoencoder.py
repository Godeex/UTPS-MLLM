import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# 添加internvl模块的路径到Python路径
# sys.path.append('/path/to/internvl/parent/directory')  # 修改为实际路径
# from internvl.model.autoencoder import AutoEncoder

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/preprocess/
project_root = os.path.join(current_dir, '../..')  # 上两级到项目根目录
sys.path.insert(0, project_root)

try:
    from internvl.model.autoencoder import AutoEncoder
    print(f"Successfully imported AutoEncoder from {project_root}")
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"Current Python path: {sys.path}")
    sys.exit(1)


def train_autoencoder(
    model,
    train_embeddings,
    num_epochs=20,
    batch_size=1024,
    patience=10,  # Early stopping patience
):
    """
    Args:
        model: 自编码器模型实例
        train_embeddings: 训练用的嵌入数据
        num_epochs: 最大训练轮数
        batch_size: 批次大小
        patience: 早停耐心值

    为特定数据集训练自编码器，学习该数据集的特征分布
    自编码器目标：输入嵌入 → 编码 → 解码 → 重建输入嵌入
    """
    # 创建数据集 - 输入和输出都是相同的嵌入向量
    dataset = TensorDataset(train_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 使用均方误差损失和Adam优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 余弦退火学习率调度 - 让学习率平滑下降
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-3
    )

    best_loss = float("inf")  # 初始化最佳损失为正无穷
    epochs_without_improvement = 0  # 记录没有改进的轮数

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data in dataloader:
            inputs = data[0].cuda()  # Ensure input is on GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        total_loss /= len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.8f}")

        # 更新学习率
        scheduler.step()

        # 早停检查
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
            break

    return model


# L2 normalization function
def l2_normalize(embeddings):
    """
    L2归一化函数

    Args:
        embeddings: 输入嵌入向量
    Returns:
        normalized_embeddings: 归一化后的嵌入向量
    """
    norm = embeddings.norm(p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norm
    return normalized_embeddings


# try to load the autoencoder model from the disk, if not exist, train and save the autoencoder
def load_or_train_autoencoder(
    folder,
    embedding,
    input_dim,
    hidden_dim,
    model_path,
    num_epochs=20,
):
    """
    加载或训练自编码器

    Args:
        folder: 数据集名称
        embedding: 嵌入数据
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        model_path: 模型保存路径
        num_epochs: 训练轮数
    """
    # 初始化自编码器并移到GPU
    autoencoder = AutoEncoder(input_dim, hidden_dim).cuda()

    if os.path.exists(model_path):
        # 如果模型已存在，直接加载
        print(f"Loading pre-trained autoencoder for {folder}...")
        autoencoder.load_state_dict(torch.load(model_path))
    else:
        # 否则训练新模型
        print(f"Training new autoencoder for {folder}...")
        autoencoder = train_autoencoder(
            autoencoder,
            embedding,
            num_epochs,
        )
        torch.save(autoencoder.state_dict(), model_path)
        print(f"Autoencoder for {folder} saved at {model_path}")

    return autoencoder


def create_reconstruction_loss_quantile_table(embeddings, autoencoders, folders):
    """create the reconstruction loss quantile table"""
    """
    创建重建损失分位数表
    了解每个任务重建损失的分布
    为get_task_ids中的阈值选择提供依据
    识别异常任务或训练问题
    
    Args:
        embeddings: 所有数据集的嵌入字典
        autoencoders: 所有自编码器字典
        folders: 数据集名称列表
    """
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    # store the results
    results = {}

    for folder in folders:
        print(f"Computing reconstruction losses for {folder}...")

        # compute the reconstruction loss
        current_losses = autoencoders[folder].compute_reconstruction_loss(
            embeddings[folder]
        )

        # compute the quantiles
        quantile_values = torch.quantile(current_losses, torch.tensor(quantiles).cuda())

        # store the results
        results[folder] = {
            f"Q{int(q*100)}": val.item() for q, val in zip(quantiles, quantile_values)
        }

        results[folder]["mean"] = current_losses.mean().item()
        results[folder]["std"] = current_losses.std().item()
        results[folder]["min"] = current_losses.min().item()
        results[folder]["max"] = current_losses.max().item()

    # convert to DataFrame
    df = pd.DataFrame(results).T

    # rearrange the columns
    columns_order = [
        "min",
        "Q10",
        "Q25",
        "Q50",
        "Q75",
        "Q90",
        "Q95",
        "Q99",
        "max",
        "mean",
        "std",
    ]
    df = df[columns_order]

    return df


def load_embeddings(folders, embeddings_dir="embeddings"):
    """
    加载所有数据集的嵌入向量

    Args:
        folders: 数据集名称列表
        embeddings_dir: 嵌入向量存储目录

    Returns:
        dict: 数据集名称到嵌入向量的映射
    """
    embeddings = {}
    for folder in folders:
        embedding_path = f"{embeddings_dir}/{folder}/embeddings.pt"
        if not os.path.exists(embedding_path):
            print(f"Warning: Embedding file not found for {folder}: {embedding_path}")
            continue

        embedding = torch.load(embedding_path).to(torch.float32).cuda()
        embeddings[folder] = l2_normalize(embedding)
        print(f"Loaded embeddings for {folder}: {embedding.shape}")

    return embeddings


def setup_autoencoders(embeddings, hidden_dim=128, base_model_dir="autoencoder_models"):
    """
    设置所有自编码器（加载或训练）

    Args:
        embeddings: 嵌入向量字典
        hidden_dim: 隐藏层维度
        base_model_dir: 模型存储基础目录

    Returns:
        dict: 数据集名称到自编码器的映射
    """
    autoencoders = {}

    for folder, embedding in embeddings.items():
        input_dim = embedding.shape[1]
        model_path = f"{base_model_dir}/{folder}/autoencoder.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # load or train the autoencoder
        autoencoder = load_or_train_autoencoder(
            folder,
            embedding,
            input_dim,
            hidden_dim,
            model_path,
            num_epochs=1000,
        )

        autoencoders[folder] = autoencoder

    return autoencoders


def main():
    """
    主函数：训练自编码器并生成重建损失分位数表
    """
    print("=== AutoEncoder Training and Analysis ===")
    embeddings_dir = "embeddings"
    # 配置参数
    folders = [
        "vizwiz_caption",
        "skvg",
        "textcaps",
        "iconqa",
        "ocrvqa",
        "flickr30k",
        "vizwiz",
        "kvqa",
        "pmcvqa",
    ]

    # Load embedding data for each task
    embeddings = {
        folder: l2_normalize(
            torch.load(f"{embeddings_dir}/{folder}/embeddings.pt").to(torch.float32).cuda()
        )
        for folder in folders
    }

    # train and save the autoencoder for each task, or load the existing autoencoder
    hidden_dim = (
        128  # the dimension of the hidden layer can be adjusted according to the need
    )
    autoencoders = {}

    for folder, embedding in embeddings.items():
        input_dim = embedding.shape[1]
        model_path = f"autoencoder_models/{folder}/autoencoder.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # load or train the autoencoder
        autoencoder = load_or_train_autoencoder(
            folder,
            embedding,
            input_dim,
            hidden_dim,
            model_path,
            num_epochs=1000,
        )

        autoencoders[folder] = autoencoder

    # create the reconstruction loss quantile table
    print("\nCreating reconstruction loss quantile table...")
    quantile_table = create_reconstruction_loss_quantile_table(
        embeddings, autoencoders, folders
    )

    # save the quantile table
    output_path = f"autoencoder_models/reconstruction_loss_quantiles.csv"
    quantile_table.to_csv(output_path)
    print(f"Reconstruction loss quantile table saved to: {output_path}")

    # print the quantile table
    print("\nReconstruction Loss Quantile Table:")
    print(quantile_table)


if __name__ == "__main__":
    main()
