# 三日 Episode / Fingerprint / Overnight A/B 最终报告

## 研究窗口

- 日期：2026-01-20、2026-01-21、2026-01-22
- 两个真实跨夜边界：01-20→01-21、01-21→01-22
- 收盘/开盘/日中窗口：各 30 个 effective states
- 社区最小规模：20 个成员
- 每层选取质量最高的 30 个稳定 episode
- 稳定 episode 要求：至少 3 个 community instances
- episode gap：0、2、5 个 effective states

## A/B 设计

- A：严格 community identity 基线（既有 Jaccard / containment）
- B：episode-aware，允许短缺口并聚合连续实例
- C1：成员 fingerprint（成员频率、核心成员、persistent members）
- C2：结构 fingerprint（规模、持续时间、实例数、gap 形态）
- C3：hybrid fingerprint（成员 + 结构）

Null controls：day-order permutation、member permutation、time-window permutation（日中→次日开盘）、fingerprint-label permutation。

当前产物缺少 point-in-time sector metadata，因此 sector-preserving null 未执行。

## 核心结果

| gap | method | actual retention | day-order retention | actual/day-order | midday/open retention | actual/midday |
|---:|---|---:|---:|---:|---:|---:|
| 0 | hybrid | 37.56% | 37.86% | 0.992 | 16.03% | 2.344 |
| 0 | structure | 68.59% | 62.14% | 1.104 | 37.31% | 1.838 |
| 2 | hybrid | 36.79% | 37.44% | 0.983 | 15.64% | 2.352 |
| 2 | structure | 69.49% | 62.74% | 1.108 | 37.56% | 1.850 |
| 5 | hybrid | 36.15% | 36.50% | 0.991 | 14.87% | 2.431 |
| 5 | structure | 68.33% | 61.79% | 1.106 | 36.79% | 1.857 |

无阈值最佳匹配排名检验中：

- raw/IDF member 的 actual vs day-order AUC 约为 0.49–0.50，接近随机。
- structure 的 actual vs day-order AUC 约为 0.46–0.47，没有真实日序优势。
- IDF hybrid 的 actual vs day-order AUC 约为 0.47–0.48，没有真实日序优势。
- structure/hybrid 的 actual vs midday→open AUC 约为 0.65–0.67，显示显著时段结构效应。

## 主要结论

1. 当前规则 fingerprint 没有解决跨夜身份问题。hybrid 的实际/day-order lift 在 gap=0/2/5 分别约为 0.992、0.983、0.991。
2. 结构 fingerprint 的 retention 较高，但成员置换后不变，说明它主要识别规模与持续形态，而不是主题身份。
3. raw/IDF member fingerprint 的真实日序与乱序日序几乎不可分，IDF 没有产生跨日身份增益。
4. 真实 close→open 明显高于 midday→open，说明存在尾盘—开盘结构状态或时段效应，但不能据此称为同一主题跨夜延续。
5. gap=2 或 gap=5 没有改善 day-order 区分度，放宽 episode 内短暂中断不会自动产生可靠跨夜 fingerprint。
6. 不能继续靠降低阈值强行提高 retention；高 retention 若同时存在于 day-order 和 member-permutation null 中，属于身份歧义。

## 下一步

- 保留 episode 作为长期数据对象。
- fingerprint 必须加入跨层轨迹、边结构统计、节点角色、行业/语义分布和收益/量能反应轨迹。
- 加入 point-in-time sector metadata 后执行 sector-preserving hard null。
- 下一轮扩展到至少一个月，以严格 point-in-time prototype 测试 recurrence。
- recurrence attention 必须使用 null-adjusted recurrence，不能只按出现次数加权。

## 判定

> Episode 有用，但仅由成员集合、规模和持续形态构成的规则 fingerprint，在三日数据上不能证明真实跨夜主题身份；当前可验证的是 close/open 时段结构效应，而不是 theme recurrence。
