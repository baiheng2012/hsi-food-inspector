"""
一维卷积神经网络 (cnn1d.py)
============================

把每个像素的光谱当作一条长度为 B 的"一维信号"，用 1D 卷积提取局部吸收峰特征。

- SpectralCNN               : 纯 PyTorch 网络（nn.Module）。之后导出 ONNX 就导出它。
- TorchSpectralClassifier   : sklearn 风格封装（fit / predict / predict_proba），
                              因此可以直接放进 Pipeline 和交叉验证，与基线模型同一套评估。
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class SpectralCNN(nn.Module):
    """
    结构：[Conv1d -> BatchNorm -> ReLU -> MaxPool] x 2 -> 展平 -> 全连接 -> 类别 logits

    输入  : (N, 1, B)  N 条光谱，1 个通道，B 个波段
    输出  : (N, n_classes) 未归一化的分数 (logits)

    设计选择：
    - 卷积核 7：一个吸收峰大约跨几个相邻波段，核太小看不到峰形。
    - 不用全局池化：吸收峰的"位置"本身就是信息，直接展平保留全部位置信息。
    - 不用 AdaptiveAvgPool1d：它在输出长度不能整除输入长度时无法导出 ONNX。
      代价：全连接层的输入维度依赖波段数 n_bands（两次 MaxPool 后为 B//2//2）。
    """

    def __init__(self, n_classes: int, n_bands: int, c1: int = 16, c2: int = 32,
                 kernel: int = 7, dropout: float = 0.3):
        super().__init__()
        pad = kernel // 2  # 保持卷积后长度不变
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel, padding=pad), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel, padding=pad), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),
        )
        flat = c2 * (n_bands // 2 // 2)  # MaxPool 向下取整
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

class TorchSpectralClassifier(ClassifierMixin, BaseEstimator):
    """
    sklearn 风格的 1D-CNN 分类器。

    参数（__init__ 只保存参数，不做任何计算，这是 sklearn 的约定，clone() 才能正常工作）
    ----
    epochs       : 训练轮数
    lr           : AdamW 初始学习率（配合余弦退火逐渐降到 0）
    batch_size   : 小批量大小
    weight_decay : 权重衰减（正则化）
    dropout      : 全连接层的 Dropout 比例
    seed         : 随机种子（权重初始化 + 数据打乱），保证可复现

    fit 之后可用的属性（sklearn 约定以下划线结尾）
    ----
    classes_      : 类别标签数组
    model_        : 训练好的 SpectralCNN
    loss_history_ : 每个 epoch 的平均训练损失（用来检查是否收敛）
    """

    def __init__(self, epochs: int = 30, lr: float = 2e-3, batch_size: int = 256,
                 weight_decay: float = 1e-4, dropout: float = 0.3, seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        # 把任意标签（如 1,2,3）映射成 0..K-1，PyTorch 的交叉熵要求如此
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        #self.model_ = SpectralCNN(len(self.classes_), dropout=self.dropout)
        self.model_ = SpectralCNN(len(self.classes_), n_bands=X.shape[1], dropout=self.dropout)        
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X)[:, None, :]            # (N, 1, B)
        yt = torch.from_numpy(y_idx.astype(np.int64))   # (N,)
        n = len(X)

        self.loss_history_ = []
        self.model_.train()  # 训练模式：启用 Dropout，BatchNorm 使用批统计量
        for _ in range(self.epochs):
            perm = torch.from_numpy(rng.permutation(n))  # 每个 epoch 重新打乱
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            self.loss_history_.append(total / n)
        self.model_.eval()  # 推理模式：关闭 Dropout，BatchNorm 使用滑动平均统计量
        return self

    def predict_proba(self, X) -> np.ndarray:
        """返回 (N, 类别数) 的概率矩阵，每行和为 1。"""
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))[:, None, :]
        outs = []
        with torch.no_grad():  # 推理不需要梯度，省内存、更快
            for i in range(0, len(X), 2048):
                logits = self.model_(X[i:i + 2048])
                outs.append(torch.softmax(logits, dim=1).numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
