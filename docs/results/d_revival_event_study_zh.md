# D 组继续探索：Revival Event Study

## 目标

不再只看 theme path 被延长多久，而是研究复燃事件本身：复燃后是否继续存活、是否扩张、是否有跨层支持，以及复燃时点后的 5/15/30/60 分钟收益。

## 事件定义

- 初始 bridge：containment >= 0.18 且 fingerprint >= 0.10。
- 主题先至少 active 3 states，才允许进入 dormant。
- dormant 最多 3 states，最多复燃 1 次。
- 基础 revival fingerprint >= 0.20。
- 真实边界：2026-01-20→21、2026-01-21→22。
- Null：三组乱序日边界。
- 对照：同层、近似规模 open-birth。

## D9–D12

- D9_event_base：所有符合基础复燃条件的事件。
- D10_postconfirm2：复燃后至少再确认 1 个状态。
- D11_breadth_confirm：D10 + revival 社区规模相对 prototype 扩张至少 10%。
- D12_crosslayer_confirm：D10 + 同时刻存在跨层支持。

D9/D10 的真实与乱序复燃事件在 3/5-state survival 和收益上没有明显分离。D12 的跨层支持几乎普遍存在，缺乏筛选力。

D11 breadth-confirm 有方向性：

- 真实 3-state survival 89.47%，乱序日 70.83%。
- 真实 5m/15m/30m 收益约 0.0728%、0.1398%、0.1881%。
- 乱序日约 0.0212%、0.0834%、0.1154%。
- 样本仅 19 vs 24，尚未显著。

## 分层候选

### D13：fingerprint >= 0.22 + breadth expansion

- 真实 3-state：86.67%，null 62.50%。
- 真实 5m：0.0921%，null 0.0461%。
- 真实 15m：0.1456%，null 0.0772%。
- 真实 30m：0.1876%，null 0.0894%。
- 方向一致，但样本 15 vs 16，未显著。

### D14：开盘前 10 states 内的 breadth revival

- 真实 3-state：86.67%，null 75.00%。
- 真实 5-state：60.00%，null 50.00%。
- 收益方向较弱，不能明显优于 open-birth。

### D15：fingerprint >= 0.30 + breadth expansion

这是目前最强候选：

- 真实样本 10，null 9。
- 真实 3-state survival：90.00%。
- null 3-state survival：44.44%。
- 差 +45.56 个百分点，p=0.0428。
- 真实平均 post-revival active hits：7.8。
- null：3.22。
- 差 +4.58 states，p=0.0394。
- 真实 5m 收益：0.1135%，null 0.0173%。
- 真实 15m：0.1650%，null 0.0434%。
- 真实 30m：0.1970%，null 0.0474%。
- 收益差方向强，但样本量下尚未达到 5% 显著。

相对实际 open-birth，D15 的 3-state survival 与收益没有显著优势；因此它目前证明的是“真实相邻日 revival 比乱序 revival 更可信”，而不是独立优于所有开盘新生主题。

## 金融解释

高 fingerprint 表示复燃社区与昨日或早前主题原型保持较强身份一致性；breadth expansion 表示复燃不是少数残余股票偶然重合，而是参与股票数量进一步扩张。两者共同出现时，更符合“隔夜信息继续被市场扩散和消化”的机制。

## 当前推荐

- D15 作为 research candidate：fingerprint >= 0.30 且 breadth expansion >= 10%。
- 只赋予 revival_attention_bonus，不直接延长持有或路径身份。
- 复燃后 3 states 内持续确认时，再升级为 confirmed_revival。
- 5/15/30m 可作为主要研究窗口；60m 没有稳定优势。
- 必须扩展到至少一个月验证，因为当前 D15 只有 10 个真实事件。

## 结论

D 的价值不在于普遍允许 dormant/revival，而在于识别少数“高指纹一致性 + breadth 扩张”的复燃事件。当前三日数据中，D15 首次在真实/乱序日之间得到显著的 3-state survival 与 post-revival 活跃时长差异，是值得扩展验证的方向。
