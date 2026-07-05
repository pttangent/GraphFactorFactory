# D 组复燃机制继续探索：先确认、后休眠、再复燃

## 目标

上一轮 D 组虽然显著延长 5/10-state survival，但 null 与 open-birth 也同步被延长，存在机械延寿。此次新增约束：主题必须先在开盘后被连续确认，才有资格进入 dormant/revival。

## 变体

- D5_confirmed：至少 3 个 active hits 后才允许 dormant；revival fp>=0.20，最多 dormant 3 states，最多复燃 1 次。
- D6_confirmed_strict：D5 + fp>=0.24、core>=0.15、size ratio 0.5–2.0、最多 dormant 2 states。
- D7_mature_bridge：D6 + 初始 bridge containment>=0.20 + revival score 高于当时截面 95% null 分位。
- D8_highfp_confirmed：初始 fingerprint>=0.14 + 跨层支持 + 截面 90% 分位。

## 关键结果

### D5_confirmed（当前最佳折中）

- 真实 5-state survival：72.88%
- day-order null：63.79%
- 差：+9.09pp，p=0.105（方向性但未显著）
- 真实 10-state：31.36%
- null 10-state：36.21%，无优势
- 真实平均复燃次数：0.534
- null：0.414，p=0.0438（真实相邻日复燃更频繁）
- 实际 5m 收益：0.0924%
- null：0.0381%，p=0.000267

相对同层同规模 open-birth：

- 5-state：72.88% vs 64.41%，+8.47pp，p=0.162
- 10-state：31.36% vs 33.05%，无优势
- 复燃次数：0.534 vs 0.441，方向性但未显著

Null 机械延寿仍未完全消失：null bridge 5-state 63.79% vs null birth 53.45%，p=0.0505。

### D6_confirmed_strict

- 真实 5-state：71.19%，null 63.22%，p=0.158
- 真实 10-state：21.19%，null 30.46%，反而更弱
- 复燃次数显著减少，但 null bridge 仍比 null birth 更容易活到 5 states（p=0.023）

### D7_mature_bridge

- 真实 5-state：70.34%，null 63.22%
- 真实 10-state：22.03%，null 30.46%
- 初始 bridge 加强和截面分位门槛没有带来真实长期优势

### D8_highfp_confirmed

- 真实 5-state：69.49%，null 63.22%
- 真实 10-state：20.34%，null 29.31%
- 高 fingerprint + 跨层确认过严，压低真实与 null，但没有改善相对排序

## 结论

1. “先确认后复燃”是正确方向，能显著降低上一版 D 的普遍机械延寿。
2. D5 是当前最佳研究参数：真实相邻日的复燃次数显著高于乱序日，5-state survival 有 +9pp 方向性优势。
3. 但 10-state 仍没有优势，说明当前数据只支持短期复燃，不支持长时间 carry-over。
4. 更严格的 D6–D8 没有改善真实/null 分离，说明单纯提高 fingerprint、core、cross-layer 阈值不是答案。
5. 当前收益优势来自 bridge 根节点本身，复燃机制并未进一步提升收益；因此 revival 只能作为 attention annotation，而不能直接作为持有延长信号。
6. 仍需加入真正 point-in-time 的质量变化、breadth expansion、层间转换方向和复燃后的增量收益，才能判断复燃是否具有独立金融价值。

## 推荐

- 保留 D5 为 research-only：min initial active hits=3，fp revive>=0.20，max dormant=3，max revivals=1。
- 生产路径继续使用 B+C：fingerprint confirmation + 3–5 state assisted persistence。
- D5 只生成 `revival_candidate` 与 `revival_attention_bonus`，不直接延长 theme path 或交易持有。
- 下一轮重点测试：复燃后 1/3/5 states 的增量 breadth、layer expansion 与 forward return，而不是继续调高相似度阈值。
