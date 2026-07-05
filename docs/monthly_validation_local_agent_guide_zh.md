# 2026-01 整月 Carry-over / Fingerprint / D15 本地 Agent 执行指南

## 1. 当前研究状态

本地 agent 若仍停留在 smoke-run 分支，需要先理解 smoke-run 阶段解决的只是：

- 数据能否读取；
- 单日/少量 snapshot 能否完成；
- 图层、社区、标签和输出路径是否连通；
- 基础时间一致性与最小回归是否通过。

Smoke run 不能回答跨夜主题是否真实、是否优于乱序日、是否优于开盘新生主题，也不能支持 D15 的统计结论。

当前应切换到：

```text
repo: pttangent/GraphFactorFactory
branch: phase1/realdata-60min-temporal-validation
```

该分支已包含三日真实数据上的：

- 严格 community / episode / fingerprint A/B；
- StockNet carry-over 三日验证；
- fingerprint-assisted persistence A/B；
- D revival event study；
- D15 候选定义。

D15 当前含义：同层主题在跨夜 carry-over 后，经过短暂 dormant，以较高 fingerprint 一致性复燃，并出现 breadth expansion。

冻结候选参数：

```text
initial bridge containment >= 0.18
initial bridge fingerprint >= 0.10
pre-revival active states >= 3
max dormant states = 3
max revivals = 1
revival fingerprint >= 0.30
breadth expansion >= 10%
post-revival confirmation >= 3 active states
```

## 2. 为什么需要整月运行

三日结果只能视为候选证据，原因包括：

- 真实跨夜边界只有 2 个；
- D15 真实事件只有约 10 个；
- 部分显著性可能由单日或单层驱动；
- 若继续在三日样本调参，会造成严重过拟合；
- 多次 A/B 中存在结果接近、或只在 5m 有优势的版本，需要在整月统一比较。

整月目标不是继续寻找更漂亮的参数，而是验证冻结候选是否稳定。

## 3. 整月 A/B 的七个验证维度

不要做所有参数的笛卡尔积。采用 4 个主臂 + 3 个消融臂，在 7 个维度上统一评估。

### 维度 1：模型臂

主臂：

- A：StockNet baseline；entry/stay containment=0.20。
- B：entry containment>=0.18 + fingerprint confirm>=0.10。
- C：B + fingerprint-assisted persistence；assisted stay containment>=0.10 且 fingerprint>=0.16，max weak gap=1。
- D15：C 的同层 revival 候选；fingerprint>=0.30 且 breadth expansion>=10%。

消融臂：

- D9：基础 revival，不要求 breadth expansion。
- D11：breadth-confirm，但不要求高 fingerprint。
- D13：fingerprint>=0.22 + breadth expansion。

合计 7 个实验臂，但主报告必须以 A/B/C/D15 为主。

### 维度 2：对照体系

- actual adjacent overnight；
- day-order permutation；
- matched open-birth；
- matched non-adjacent same-weekday；
- volatility/open-gap matched null（若元数据可用）。

### 维度 3：时间持续性

- 3-state survival；
- 5-state survival；
- 10-state survival；
- post-revival active hits；
- dormant gap states；
- calendar gap minutes。

### 维度 4：收益时间尺度

- 5m；
- 15m；
- 30m；
- 60m；
- 若标签可用再增加 1m、3m、10m。

### 维度 5：组合口径

- equal-weight members；
- core members；
- top-5 core members；
- excess return vs SPY/QQQ；
- market-neutral return（若 benchmark label 可用）。

### 维度 6：图层分层

所有 path / episode / fingerprint / revival 必须同层追踪。分别报告每个 layer，禁止把跨层支持作为 D15 必要条件。

### 维度 7：稳健性

- leave-one-day-out；
- 每日贡献；
- 单层贡献；
- event concentration；
- bootstrap confidence interval；
- multiple-testing adjusted p-value。

## 4. 运行原则

1. 参数冻结：整月主运行中不得根据中途结果修改主臂阈值。
2. Point-in-time：prototype 只能使用事件发生前已知状态，禁止未来信息回填。
3. 同层身份：layer theme、path、episode、fingerprint、revival 均限制在同层。
4. 全漏斗保存：成功和失败事件都必须输出。
5. 原子任务：最小任务单元为 `overnight_boundary × experiment_arm`。
6. 幂等：同一 task_id 重跑必须得到同一结果；已完成任务默认跳过。
7. 不覆盖：任何参数、代码 commit 或数据版本变化必须生成新的 run_id。
8. 不删除原始中间产物；最终聚合只能读取已校验完成的 shard。

## 5. 中间产物与目录

推荐目录：

```text
outputs/monthly_carryover_ab/
  run_manifest.json
  task_index.parquet
  checkpoints/
    <task_id>.json
  shards/
    date_from=YYYY-MM-DD/date_to=YYYY-MM-DD/arm=A/
      bridge_candidates.parquet
      path_states.parquet
      revival_events.parquet
      matched_controls.parquet
      outcomes.parquet
      task_manifest.json
      _SUCCESS
  aggregates/
    group_summary.parquet
    effect_tests.parquet
    layer_summary.parquet
    daily_summary.parquet
    leave_one_day_out.parquet
  logs/
    runner.jsonl
    errors.jsonl
```

每个 task shard 必须先写入临时目录：

```text
.tmp/<task_id>/
```

完成后执行：

1. schema 校验；
2. 行数和主键唯一性校验；
3. 文件 SHA256；
4. 写 `task_manifest.json`；
5. 原子 rename 到正式 shard；
6. 最后写 `_SUCCESS`。

只有存在 `_SUCCESS` 且 manifest 校验通过的任务，才视为完成。

## 6. 断点续传

启动时读取 `task_index.parquet` 和 `checkpoints/*.json`：

- `_SUCCESS` 存在且校验通过：skip；
- 临时目录存在但无 `_SUCCESS`：清理临时目录后重跑；
- 正式目录存在但 checksum 不符：移入 `corrupt/` 后重跑；
- 失败任务：写 errors.jsonl，并继续其他任务；
- 支持 `--retry-failed` 只重跑失败任务；
- 支持 `--only-boundary`、`--only-arm`、`--from-task-id`。

Checkpoint 必须包含：

```json
{
  "task_id": "...",
  "status": "pending|running|success|failed",
  "repo_commit": "...",
  "data_version": "...",
  "config_hash": "...",
  "started_at": "...",
  "finished_at": "...",
  "output_files": [],
  "row_counts": {},
  "sha256": {},
  "error": null
}
```

## 7. Task 规划

若 2026-01 有 N 个交易日，则真实 overnight boundary 为 N-1 个。

每个真实 boundary 运行 7 个臂；null 不要全部笛卡尔展开，采用固定种子抽样：

- 每个真实 boundary：3 个 day-order null；
- 1 个 matched open-birth control；
- 可选 1 个 volatility/open-gap matched null。

建议任务总量约为：

```text
(N-1) × 7 主/消融臂 × (1 actual + 3 day-order null)
```

open-birth control 在每个 actual/null task 内部生成，不另起完整任务。

## 8. 月度验收标准

D15 只有同时满足以下条件，才升级为稳定候选：

- actual vs day-order null 的 3-state survival 显著更高；
- post-revival active hits 显著更高；
- 至少一个主要图层稳定；
- leave-one-day-out 后方向一致；
- 不由单日贡献超过 40%；
- 至少在 5m 或 15m 收益上优于 null；
- 相对 matched open-birth 至少一个持续性或收益指标有稳定优势；
- multiple-testing 校正后仍保留核心结论，或效应量足够大且置信区间稳定。

若只优于 day-order null、但不优于 open-birth，则定位为 carry-over prior，而非独立 alpha。

## 9. 本地 Agent 操作步骤

```bash
git fetch origin
git checkout phase1/realdata-60min-temporal-validation
git pull --ff-only origin phase1/realdata-60min-temporal-validation
python -m pip install -e .
```

先检查：

```bash
python scripts/run_carryover_fingerprint_ab.py --help
python scripts/run_monthly_carryover_ab.py --config configs/monthly_carryover_ab_2026_01.json --dry-run
```

正式运行：

```bash
python scripts/run_monthly_carryover_ab.py \
  --config configs/monthly_carryover_ab_2026_01.json \
  --resume
```

失败重试：

```bash
python scripts/run_monthly_carryover_ab.py \
  --config configs/monthly_carryover_ab_2026_01.json \
  --resume --retry-failed
```

## 10. 禁止事项

- 不要回到 smoke-run 分支继续整月研究；
- 不要在运行中途根据结果修改 D15；
- 不要把跨层支持当成同一主题身份确认；
- 不要只保存显著结果；
- 不要只输出汇总表而丢失事件级明细；
- 不要在没有 `_SUCCESS` 和 checksum 的情况下把 shard 纳入聚合；
- 不要因为单月结果不显著就覆盖或删除失败产物。
