# TAAC-2026 HyFormer Baseline

这个仓库公开的是我们一版最终整理出来的 HyFormer 训练代码。这里主要记录两部分内容：

1. 当前公开代码相对原始 baseline 的实际改动。
2. 过去实验中验证过有效的改动方向。

## 当前公开代码的主要改动

当前公开版的核心改动主要在 `user dense` 分组建模上，主干仍然是多序列 HyFormer。

### 1. user dense 不再整体压成一个 token

原始 baseline 里，user dense 基本是直接拼接后压成一个 token。当前版本把它按特征形态拆开处理，再融合成一个最终的 user dense token。

### 2. 各组 dense 的具体处理方式

#### `fid 61`

- 单独取出
- 做 L2 normalize
- 过 `Linear + LayerNorm + SiLU`
- 再接一个轻量 residual FFN

#### `fid 87`

- 单独取出
- 做 L2 normalize
- 过 `Linear + LayerNorm + SiLU`
- 再接一个轻量 residual FFN

#### `fid 89/90/91`

- 作为一组趋势型 dense 处理
- 先按 `3 x L` 的形式拼起来
- 送进 `QuantileTrendConvEncoder`
- 输出后再接一个轻量 residual FFN

这部分的直觉是把它们当成一组有顺序结构的趋势信号，而不是普通拼接 dense。

#### `fid 62/63/64/65/66`

- 作为一组变长 dense 单独处理
- 每个 fid 先取出对应 dense
- 先做 `clamp_min(0)`，再做 `log1p`
- 再做 L2 normalize
- 对每个 fid 额外提取两个统计量：
  - 均值
  - L2 norm
- 把所有归一化后的向量和这些统计量拼起来
- 过 `Linear + LayerNorm + SiLU`
- 再接一个轻量 residual FFN

这部分是当前公开版里最重要的 dense 增强，因为 `62-66` 这组特征明显不是普通 dense，直接拼接或平均都不够合理。

#### request time 离散时间特征

- 单独编码 hour / weekday / day_of_month / part_of_day
- embedding 后拼接
- 过 `Linear + LayerNorm + SiLU`
- 再接一个轻量 residual FFN

#### request time sin/cos 周期特征

- 单独取 8 维周期特征
- 过 `Linear + LayerNorm + SiLU`
- 再接一个轻量 residual FFN

### 3. 最终融合方式

上面这些 branch 不各自作为独立 token 输出，而是先分别编码成 `d_model` 向量，再拼接后通过一层 `Linear + LayerNorm + SiLU`，最终融合成 **1 个 user dense token**。

也就是说，当前版本不是扩大 dense token 数，而是先做结构化分组，再把信息压回一个更干净的 dense token。

### 4. EMA

当前版本保留了 EMA：

- 训练时对 dense 参数维护 shadow weights
- 验证时使用 EMA 权重
- 选择 best model 时也使用 EMA 权重

## 过去我们尝试过且验证有效的改动

下面这些是过去实验中验证过有效的方向，不一定全部包含在当前公开仓库里。

### 1. `action_type` 调制序列 token

- 让序列 token 显式感知 action_type
- 用 gate + FiLM 的方式调制 sequence token
- 属于对序列语义的增强，不改 HyFormer 主干

### 2. time-aware stat tokens

- 给 user/item 标量离散特征构造历史频次统计
- 给 user/item 标量离散特征构造历史 CVR 统计
- 统计按时间滚动生成，避免未来信息泄漏
- 再把这些统计编码成 token，放进 NS 侧参与交互

### 3. paired 特征单独建模

- `62-66` 这类变长 paired 特征单独处理有效
- `89-91` 这类固定槽位 paired 特征单独处理有效
- 明显优于把它们混进普通 int / dense 路径

### 4. standalone dense 拆开处理

- `61` 和 `87` 分开建模有效
- 比直接把所有 dense 压成一个 token 更稳

### 5. request time 特征有效

- request time 的离散时间特征有效
- request time 的 sin/cos 周期特征有效
- 序列行为相对请求时间的 time bucket 也有效

### 6. NS 侧做更细粒度拆分有效

- 不同形态的 dense / paired / trend 特征分开处理有效
- 比“所有非序列特征粗暴压缩”效果更好

### 7. query 数增加是有效方向

- 对四个长序列 domain，增加 query 数通常是正向改动
- 本质上是增加从序列里读取信息的视角

### 8. user-item cross token 是有效方向

- 在 NS 侧显式加入 user-item 交互 token 是有效的
- 属于低成本增强 matching 信号的方法

### 9. valid finetune / 后处理式训练策略有收益

- 在 best checkpoint 基础上再做 valid finetune 是有效方向
- 更适合最终冲榜，不适合作为普通离线验证结论

## 当前公开仓库没有完整带出的方向

下面这些方向我们验证过有效，但当前公开仓库没有完整保留：

- `action_type` 调制 sequence token
- time-aware stat tokens
- user-item cross token
- 更大的 NS token 预算重分配
- valid finetune
