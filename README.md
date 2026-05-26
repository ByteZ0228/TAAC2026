# TAAC-2026 HyFormer Baseline

这个仓库公开的是我们一版最终整理出来的 TAAC2026 代码。这里主要记录两部分内容：
1. 当前公开代码相对原始 baseline 的实际改动。
2. 过去实验中验证过有效的改动方向。

我们最终的成绩是0.830。除了已经写在这里的思路，我们另外也尝试过很多其他的思路了，不过很多很合理的方向或者开源的上分思路套到我们当前的模型上是无效的，加上比赛的测试波动也比较大，小改动不好做消融，因此还有许多改动无从知道是否有效。我们队伍在很长一段时间内都在纠结应该保存多少token，我们在20token的方案上做了将近2个星期都卡在0.829无法提升。此时发现前排token数几乎都是8个左右，重新制定策略后很多之前验证有效的策略都无法使用了，比如说fid的key的统计ueie交叉，action_type，rankloss，SENet等等在我们最终的改动上都没有办法取得收益，也没有充足的时间进行进一步尝试。不过第一次打推荐比赛，学习到了很多知识也看了不少论文，交流群里大家讨论的氛围也非常浓厚，学习到了很多东西，明年继续加油。

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

### 5. query 生成不是简单 mean pooling

原始文档版 baseline 里，query 生成更接近“对每个序列做摘要，再生成 query”。当前公开代码里，这一步已经不是简单 mean pooling，而是更接近 DIN 风格的 target-aware weighted pooling：

- 先对 item tokens 做平均，得到 `item context`
- 对每个 sequence token，用
  - `seq`
  - `item_ctx`
  - `seq * item_ctx`
  - `seq - item_ctx`
  这几项拼接后打分
- 对历史序列位置做 softmax 加权
- 再做加权求和，得到当前 domain 的 sequence summary

也就是说，这里已经从“无目标的均值摘要”变成了“item-aware 的序列加权摘要”。

### 6. block 里有 NS-conditioned domain gate

在每个 `MultiSeqHyFormerBlock` 开头，当前代码会先根据各个序列 domain 的上下文，对 NS tokens 做一次动态调制：

- 先对 NS tokens 做全局摘要
- 再分别对每个 domain 的序列做摘要
- 根据 `ns_global` 和 `domain_ctx` 计算每个 domain 的权重
- 得到一个融合后的 `domain context`
- 再把这个 context 投回 NS tokens

这个改动的作用是让 NS 侧在进入后续 sequence evolution / query decoding 之前，先感知当前样本里哪些 domain 更重要。

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

### 8. DIN 风格 target-aware pooling 是有效方向

- 用 item context 对历史序列位置做加权通常优于简单 mean pooling
- 本质上是让 query 生成阶段更早感知 target item

### 9. user-item cross token 是有效方向

- 在 NS 侧显式加入 user-item 交互 token 是有效的
- 属于低成本增强 matching 信号的方法

### 10. domain gate 是有效方向

- 先根据各个 domain 的上下文，动态调制 NS tokens 是有效方向
- 本质上是在 block 内增加一次跨 domain 的轻量条件路由

