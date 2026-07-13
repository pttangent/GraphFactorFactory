# 月度 Alpha 评估报告 (2026-01)

## 1. 概览 (Overview)

- **评估阶段**: P0 (图边特征提取)
- **输入文件数**: 40
- **截面回归评估次数**: 3500385 次
- **对齐契约**: `p2-pit-v2`

## 2. 顶级因子表现 (Top Rank IC)

这是所有特征中，日内预测表现最好的 Top 10（基于平均 Rank IC 的绝对值）：

| Feature | Target | Layer | Scale | Rank IC | Spread | Win Rate |
|---------|--------|-------|-------|---------|--------|----------|
| `p0_edge_spillover_signal` | `target_15m` | 1 | default | -0.0156 | -0.000425 | 26.04% |
| `p0_edge_spillover_sum` | `target_15m` | 1 | default | -0.0155 | -0.000422 | 25.22% |
| `p0_edge_spillover_signal` | `target_30m` | 1 | default | -0.0148 | -0.000436 | 30.06% |
| `p0_edge_spillover_sum` | `target_30m` | 1 | default | -0.0147 | -0.000430 | 30.82% |
| `p0_edge_spillover_signal` | `target_15m` | 14 | default | -0.0115 | -0.000284 | 32.19% |
| `p0_edge_spillover_sum` | `target_15m` | 14 | default | -0.0112 | -0.000279 | 32.42% |
| `p0_edge_spillover_signal` | `target_30m` | 14 | default | -0.0111 | -0.000297 | 35.35% |
| `p0_edge_spillover_signal` | `target_60m` | 1 | default | -0.0109 | -0.000339 | 36.90% |
| `p0_edge_spillover_sum` | `target_60m` | 1 | default | -0.0109 | -0.000340 | 36.80% |
| `p0_edge_spillover_sum` | `target_30m` | 14 | default | -0.0108 | -0.000292 | 36.76% |

## 3. 按层级分类最佳特征

### Layer: 1
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_signal` | `target_15m` | default | -0.0156 |
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0155 |
| `p0_edge_spillover_signal` | `target_30m` | default | -0.0148 |

### Layer: 2
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0016 |
| `p0_edge_spillover_signal` | `target_15m` | default | -0.0014 |
| `p0_edge_spillover_sum` | `target_30m` | default | -0.0014 |

### Layer: 3
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0016 |
| `p0_edge_spillover_signal` | `target_15m` | default | -0.0015 |
| `p0_edge_spillover_sum` | `target_30m` | default | -0.0013 |

### Layer: 4
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0081 |
| `p0_edge_spillover_signal` | `target_15m` | default | -0.0077 |
| `p0_edge_spillover_sum` | `target_30m` | default | -0.0074 |

### Layer: 5
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_abs_weight` | `target_120m` | default | 0.0091 |
| `p0_edge_mean_abs_weight` | `target_120m` | default | 0.0089 |
| `p0_total_weight_sum` | `label_120m` | default | 0.0073 |

### Layer: 6
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_total_weight_sum` | `label_30m` | default | 0.0016 |
| `p0_total_edge_count` | `label_120m` | default | 0.0016 |
| `p0_total_weight_sum` | `label_120m` | default | 0.0015 |

### Layer: 7
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_mean_abs_weight` | `target_120m` | default | 0.0081 |
| `p0_total_weight_sum` | `label_120m` | default | 0.0075 |
| `p0_edge_abs_weight` | `target_120m` | default | 0.0060 |

### Layer: 8
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_30m` | default | -0.0017 |
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0016 |
| `p0_edge_spillover_sum` | `target_60m` | default | -0.0016 |

### Layer: 9
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0017 |
| `p0_edge_spillover_sum` | `target_60m` | default | -0.0015 |
| `p0_edge_spillover_sum` | `target_30m` | default | -0.0014 |

### Layer: 10
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_120m` | default | -0.0013 |
| `p0_edge_count` | `target_120m` | default | 0.0012 |
| `p0_edge_abs_weight` | `target_120m` | default | 0.0012 |

### Layer: 11
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_sum` | `target_30m` | default | -0.0016 |
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0013 |
| `p0_edge_spillover_signal` | `target_30m` | default | -0.0013 |

### Layer: 12
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_mean_abs_weight` | `target_15m` | default | 0.0012 |
| `p0_edge_mean_abs_weight` | `target_120m` | default | 0.0009 |
| `p0_edge_mean_abs_weight` | `target_30m` | default | 0.0008 |

### Layer: 13
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_mean_abs_weight` | `target_120m` | default | 0.0026 |
| `p0_edge_abs_weight` | `target_120m` | default | 0.0022 |
| `p0_edge_mean_abs_weight` | `target_60m` | default | 0.0019 |

### Layer: 14
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_spillover_signal` | `target_15m` | default | -0.0115 |
| `p0_edge_spillover_sum` | `target_15m` | default | -0.0112 |
| `p0_edge_spillover_signal` | `target_30m` | default | -0.0111 |

### Layer: 15
| Feature | Target | Scale | Rank IC |
|---------|--------|-------|---------|
| `p0_edge_mean_abs_weight` | `target_120m` | default | 0.0063 |
| `p0_edge_mean_abs_weight` | `target_60m` | default | 0.0048 |
| `p0_edge_mean_abs_weight` | `target_30m` | default | 0.0039 |

## 4. 后续流水线

P0 图边评估已在 12 核并发下于 25 分钟内顺利完成。目前流水线正在跳步进行 P1 (Theme Relation) 和 P2 的评估，敬请期待完整的月度终极报告。
