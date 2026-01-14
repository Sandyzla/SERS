# 模型架构深度解析 (模型参数有变动，仅供参考)

本文档旨在详细解析定义在 `src/SERS/main.py` 中的 **CoreModel** 模型。这是一个典型的 **混合模型 (Hybrid Model)**，结合了 **CNN**（擅长提取局部特征）和 **Transformer**（擅长捕捉全局上下文依赖）。

为了方便理解，本文假设以下常量设置：
* **BATCH_SIZE:** 32
* **INPUT_LEN:** 1201 (序列长度/波长点数量)

---


## 1. 输入层 (Input)
* **输入数据:** 一个 Batch 的 SERS 光谱数据。
* **张量形状 (Tensor Shape):** `(32, 1, 1201)`
    * `32`: Batch Size (一次训练32个样本)。
    * `1`: Channel (单通道，类似于灰度图，这里是单条光谱强度)。
    * `1201`: Sequence Length (波长点的数量)。

## 2. 可学习的光谱掩码 (Learned Spectral Mask)
这是模型的第一层，作用类似于“注意力门控”，让模型自己学习哪些波长重要，哪些是噪声。

* **代码对应:** `self.spectral_mask`
* **参数:** `self.mask_weights`，形状为 `(1, 1, 1201)` 的可训练参数。
* **操作:** `x * self.mask_weights` (元素级乘法)。
* **原理:**
    * 如果模型发现第 800 个波长点是噪声，它会将对应权重学成接近 **0**。
    * 如果第 1000 个点是特征峰，权重会变大。
    * **L1 正则化:** 训练循环中加入了 `L1_LAMBDA * l1_loss`，强迫 mask 变得稀疏（大部分是0），起到**特征选择**的作用。
* **输出形状:** `(32, 1, 1201)` (数值被加权)。

## 3. CNN 前端 (局部特征提取)
负责从光谱中提取“形状特征”（如峰的尖锐程度、峰宽、局部斜率）。

### 3.1 多尺度卷积块 (Multi-Scale Block)
针对 SERS 光谱中不同宽度的峰（尖峰 vs 鼓包）设计。
* **输入:** `(32, 1, 1201)`
* **三个分支 (Parallel Branches):**
    1.  **小核 (k=3):** 感受野小，捕捉极窄的尖峰 (padding=1)。
    2.  **中核 (k=7):** 感受野中等，捕捉常规特征峰 (padding=3)。
* **融合 (Fusion):**
    * 拼接 (Concat): 2个分支（每支8通道） $\rightarrow$ 16通道。
    * 1x1 卷积: 将通道数从 16 降回 8。
* **输出形状:** `(32, 8, 1201)`

### 3.2 第一次下采样 (Pooling 1)
* **操作:** `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(kernel=2, stride=2)`
* **长度计算:** $\lfloor 1201 / 2 \rfloor = 600$
* **输出形状:** `(32, 8, 600)`

### 3.3 第二层卷积 (Conv Layer 2)
* **操作:** `Conv1d(8, 32, kernel_size=5, padding=2)`
* **目的:** 增加通道数（8变32），提取更抽象的特征。
* **输出形状:** `(32, 32, 600)` (长度不变)。

### 3.4 第二次下采样 (Pooling 2)
* **操作:** `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)`
* **长度计算:** $600 / 2 = 300$
* **输出形状:** `(32, 32, 300)`

### 3.5 第三层卷积 (Conv Layer 3)
* **操作:** `Conv1d(32, 64, kernel_size=3, padding=1)`
* **目的:** 输入 Transformer 前的最终特征提取。
* **输出形状:** `(32, 64, 300)`

## 4. 维度置换 (Permute) —— 关键步骤
CNN 和 Transformer 对数据形状的要求不同，此处进行转换。

* **CNN 格式:** `(Batch, Channel, Length)` $\rightarrow$ `(32, 64, 300)`
* **Transformer 格式:** `(Batch, Sequence_Length, Embedding_Dim)`
    * 将 `300` 视为序列长度（如同句子中的单词数）。
    * 将 `64` 视为每个点的特征向量维度。
* **操作:** `x.permute(0, 2, 1)`
* **输出形状:** `(32, 300, 64)`

## 5. 位置编码 (Positional Embedding)
Transformer 本身不具备“顺序”概念，需注入位置信息。

* **参数:** `self.pos_embedding` 形状 `(1, 500, 64)`。
* **操作:** 截取前 300 个位置向量并相加：`x = x + self.pos_embedding[:, :300, :]`。
* **输出形状:** `(32, 300, 64)` (数值包含位置信息)。

## 6. Transformer 编码器 (Encoder)
负责分析光谱的全局上下文（例如：“如果在位置 50 有个峰，且位置 200 没有峰，那大概率是 CV”）。

* **配置:** 2层 (`num_layers=2`)。
* **内部结构:**
    * **Multi-Head Self-Attention:** `d_model=64`, `nhead=4`。将特征切分为4个头，计算300个点之间的两两关系矩阵。
    * **FeedForward Network:** 线性层 `64` $\rightarrow$ `256` $\rightarrow$ `64`。
    * **Add & Norm:** 残差连接和层归一化。
* **输出形状:** `(32, 300, 64)` (形状保持不变)。

## 7. 分类器 (Classifier)
将深层特征转化为 3 个类别的概率。

### 7.1 展平 (Flatten)
* **操作:** 把序列长度和特征维度拉直。
* **计算:** $300 \times 64 = 19,200$。
* **输出形状:** `(32, 19200)`

### 7.2 全连接层 (MLP)
* **Linear 1:** `19200` $\rightarrow$ `128`
    * *激活:* ReLU & Dropout (防止过拟合)。
    * *形状:* `(32, 128)`
* **Linear 2 (Output):** `128` $\rightarrow$ `3` (3类：CV, MG, MB)
    * *形状:* `(32, 3)`

---

## 🚀 改进建议 (Optimization Suggestions)

基于上述架构分析，以下是几个潜在的痛点和改进方案：

### 1. 参数量爆炸问题 (Flatten 层)
* **问题:** Flatten 产生了 19,200 维的向量，导致第一个全连接层参数巨大 ($\approx 245$万参数)。容易过拟合且计算重。
* **改进:** 使用 **全局平均池化 (Global Average Pooling, GAP)**。
    * 在分类前对序列维度 (300) 求平均。
    * 变换: `(32, 300, 64)` $\rightarrow$ `(32, 64)`。
    * *效果:* 参数量减少 300 倍，模型更轻量稳定。

### 2. 位置编码的局限
* **问题:** `self.pos_embedding` 固定最大长度 500。如果光谱输入长度改变（如更换光谱仪），代码会报错。
* **改进:** 使用 **正弦/余弦位置编码 (Sinusoidal Position Encoding)** 或在 `forward` 中进行动态插值。

### 3. Transformer 的作用域 (分辨率)
* **问题:** Transformer 处理的是下采样 4 倍后的特征 (长度 300)，微小的峰位移可能丢失。
* **改进:**
    * 减少 CNN 的 Pooling 次数。
    * 使用 **空洞卷积 (Dilated Convolution)** 代替 Pooling，在保持序列长度的同时扩大感受野。

### 4. Token 定义 (ViT 思路)
* **现状:** 目前每个 Token 代表 CNN 提取的特征块。
* **改进:** 尝试 **Spectral ViT**。将原始光谱切分成 Patch（例如每 16 个波长切一段），直接线性映射为 Token，完全去掉 CNN 前端。（注：通常需要更大的数据量支持）。
