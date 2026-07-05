# StockNet Carry-over + Fingerprint A/B（三日严格版）

## 设计

- A：StockNet baseline，entry/stay containment=0.20。
- B：entry containment>=0.18，且 fingerprint>=0.10 才确认。
- C：B + fingerprint-assisted persistence；stay 可降至 containment>=0.10，但 fingerprint>=0.16，最多允许 1 个 weak gap。
- D：C + dormant/revival；最多 dormant 3 states，revival fingerprint>=0.20。
- 真实边界：2026-01-20→21、01-21→22。
- Null：乱序日边界；另有同层同规模 open-birth control。

## 结果摘要

### A：StockNet baseline

- 实际 3-state survival 59.15%，day-order null 43.27%，差 +15.89pp，p=0.0398。
- 实际 5m 平均收益 0.0996%，null 0.0446%，p=0.00093。
- 10-state 没有优势。

### B：Fingerprint confirmation

- 保留 26 个真实 bridge、41 个 null bridge。
- 真实 3-state survival 57.69%，null 39.02%；方向改善但样本量下 p=0.140。
- 真实 5m 收益 0.1126%，null 0.0090%，p=0.00198。
- 15m 以后优势反转，因此 B 更像短时开盘筛选器。

### C：Fingerprint-assisted persistence

- 实际 3-state 84.62%，5-state 76.92%，但 10-state 仅 7.69%。
- 相对 open-birth：3-state +23.08pp，p=0.0649；5-state +15.38pp，未显著。
- 相对 day-order：3-state +13.88pp、5-state +13.51pp，但均未显著；10-state 反而更弱。
- 说明 C 能合理延长短期确认，但不能支持长期 carry-over。

### D：Dormant/revival

- 实际 5-state 100%，null 80.49%，p=0.0178。
- 实际 10-state 88.46%，null 63.41%，p=0.0258。
- 相对实际 open-birth：5-state +26.92pp，p=0.0051；10-state +30.77pp，p=0.0137。
- 但 null bridge 相对 null open-birth 也被明显延长，说明 revival 机制存在普遍延寿偏差。

## 判定

1. B 是当前最稳健的增强：降低样本量，但显著提高 5m 真实/null 收益分离。
2. C 有短期延续价值，适合将确认窗口延长到 3–5 states，不应延到 10 states。
3. D 显示 fingerprint revival 有潜力，但必须加入 null-adjusted penalty、质量/跨层确认，否则会机械延寿。
4. 当前推荐生产候选：B 作为 bridge filter，C 作为最多 3–5 states 的 weak persistence；D 仅研究模式。
5. 收益优势集中在 5m，15m 后消失或反转，金融上更像隔夜信息在开盘的短时继续消化，而不是日内长趋势。

## 推荐参数

```text
entry containment >= 0.18
fingerprint confirm >= 0.10
normal stay containment >= 0.20
assisted stay containment >= 0.10 AND fingerprint >= 0.16
max weak gap = 1
confirmation horizon = 3–5 effective states
revival = research-only; fingerprint >= 0.20, max dormant = 3
```
