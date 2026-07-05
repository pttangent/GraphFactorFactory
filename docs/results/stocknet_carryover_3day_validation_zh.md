# StockNet 跨日 Carry-over 在 GFF 上的三日严格验证

## 目标

将 StockNet 的连续路径思想迁移到 GFF：以前一交易日最后一个有效状态作为下一交易日开盘先验，并在开盘后继续追踪，区分 bridge-only、open-confirmed 与 post-open-persistent。

## 数据与方法

- 日期：2026-01-20、2026-01-21、2026-01-22
- 真实跨夜边界：01-20→01-21、01-21→01-22
- day-order null：01-20→01-22、01-22→01-21、01-21→01-20
- 最小社区规模：20
- 开盘后追踪：30 个 effective states
- StockNet 相似度：成员 containment = |A∩B| / min(|A|,|B|)
- 阈值扫描：0.15、0.20、0.25、0.30；原 StockNet 默认 0.50 也检查
- 开盘新生对照：每个 bridge 匹配一个同层/近似规模的 unmatched open community
- 结果指标：3/5/10/20-state 生存率、平均生存状态数、5/15/30/60/120m 未来成员平均收益

## 重要方法映射

StockNet 的生命周期对象是跨层 consensus theme；GFF 当前输入是单层 community instance。因此：

- 直接使用 StockNet 默认 containment=0.50，真实与乱序边界均为 0 bridge。
- 两个真实跨夜边界的最大单层 community containment 分别约为 0.25 和 0.30。
- 本轮采用跨层候选匹配来近似 consensus carry-over，并用阈值扫描确认可行区间。

## 主要结果

### containment = 0.20（最有解释力的折中）

- 真实 bridge：72
- day-order null bridge：104
- 真实 bridge 至少存活 3 states：58.33%
- null bridge 至少存活 3 states：41.35%
- 差异：+16.99 个百分点，p=0.027
- 真实 bridge 至少存活 5 states：44.44%
- null bridge 至少存活 5 states：34.62%
- 差异：+9.83 个百分点，但 p=0.190
- 10-state 生存率：真实 1.39%，null 4.81%，无优势
- 20-state 生存率：均为 0

与同层同规模 open-birth 对照比较：

- 3-state：58.33% vs 56.94%，无显著差异
- 5-state：44.44% vs 44.44%，无差异
- 平均生存状态数：3.97 vs 4.22，无优势

未来收益：

- 真实 bridge 相对 null bridge 的 5m 平均收益更高（0.1020% vs 0.0444%，p=0.00056）
- 15m、30m、60m、120m 没有持续正向优势；15m 以后多数低于 null
- 与 open-birth 对照相比，各期限收益均无显著优势

### containment = 0.15

- 真实 bridge：258
- null bridge：379
- 真实 3-state 生存率：70.93%
- null 3-state 生存率：62.01%，p=0.020
- 真实 5-state 生存率：63.57%
- null 5-state 生存率：55.15%，p=0.034
- 但与 open-birth 对照无显著优势
- 阈值过宽，主题身份歧义明显

### containment = 0.25

- 真实 bridge：7
- null bridge：24
- 样本太少，无法可靠判断

### containment = 0.30

- 真实 bridge：1
- null bridge：3
- 不具统计解释力

### containment = 0.50

- 真实 bridge：0
- null bridge：0
- 对 GFF 单层 community 过严，不能直接照搬 StockNet 默认阈值

## 判定

1. StockNet carry-over 思想在 GFF 上有尝试价值。
2. 真实相邻日 bridge 在开盘后短期确认（3–5 states）上，确实比乱序日 bridge 更强；这支持“昨日主题作为今日开盘先验”的解释。
3. 但 bridge 尚未优于同层同规模 open-birth，因此 carry-over 目前不是独立充分信号。
4. 它最适合作为 attention prior：昨日尾盘已出现 + 今日开盘匹配 + 开盘后继续 3–5 states 时提高权重。
5. 当前没有证据支持长时间持续：10-state 以后优势消失。
6. 0.20 是当前最合理的候选阈值；0.15 过宽，0.25 以上样本不足，0.50 不适配单层 GFF community。

## 推荐实现

新增状态：

- `overnight_bridge`
- `open_confirmed_3`
- `open_confirmed_5`
- `post_open_persistent_10`
- `carryover_failed`

建议分数：

`carryover_attention = bridge_score × confirmation_3_5 × quality × breadth_expansion × null_adjustment`

其中 bridge 本身只作为先验，必须由开盘后 3–5 effective states 的持续性确认。

## 下一步

- 先构建真正的跨层 consensus theme，再复测 StockNet 默认 0.50。
- 将 node centrality、跨层支持、收益/量能扩散加入确认条件。
- 扩展到至少一个月，检验 0.20 阈值是否稳定。
- 单独测试 confirmed carry-over 对 leader/follower 和 breadth expansion 的预测力。
