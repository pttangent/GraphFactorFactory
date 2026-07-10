# P1 B50/B35 Pipeline Performance Analysis

## ?象 (Symptoms)
- **极度?慢**：?日?据（例如 2026-01-02）在??程下耗?接近 **3 小?**。按此推算 124 天需要 15 天以上才能跑完。
- **?存爆炸 (OOM)**：如果使用多核并?（例如 --workers 2 或 8），哪怕是?理??含有 1.49 ???的极端交易日，峰值?存也?瞬?拉爆 128GB 的物理?存，?出 numpy._core._exceptions._ArrayMemoryError: Unable to allocate 25.5 GiB 的底???。

## 核心瓶??? (Structural Bottlenecks)

通??查 scripts/build_b50_b35_theme_forest.py 的源?，暴露出以下?重拖慢速度且极度吃?存的架构缺陷：

### 1. 暴力全量?存加? (Memory Bloat & Zero Chunking)
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
# ... followed by ...
out = df.rename(...).copy()
out = out[cols]
out = out.dropna(...)
out = out[out["src_id"] != out["dst_id"]].copy()

**??**：?本毫?忌?地把?日期下**所有** Layer 和 Scale 的 Parquet 分片文件一次性?入?一的 Pandas DataFrame。在 1.49 ????的日子，??在?存中?生 4 到 5 次全量拷?。?正是触? 25.5 GiB ?次?存?申??而直接 OOM 的罪魁?首。

### 2. ? Python ?象与字典遍? (Inefficient Native Types)
def build_adj(edges: pd.DataFrame, members: set[int] | None = None) -> dict[int, dict[int, float]]:
    for r in edges.itertuples(index=False):
        ...

**??**：?于上???的?据集，使用 itertuples() ?每一行??命名元?，并且使用 Python 原生的嵌套 dict 构建?接矩?（Adjacency Matrix）极其致命。?不??生了天文??的?碎?存碎片，而且完全?失了 Pandas/Numpy 向量化?算的优?，速度慢了至少 100 倍。

### 3. 未加利用的?据集分? (Missing GroupBy Vectorization)
def build_relation_edges(...) -> list[dict]:
    for r in edges.itertuples(index=False):
        ...
        inter[(x, y)] += w
        counts[(x, y)] += 1

**??**：在构建子?（B50/B35）主???系??，?本又?行了一次上???的原生 Python 循?遍?和字典加法。?一步完全可以通? Pandas 的 groupby(["src_theme_id", "dst_theme_id"]).agg({"weight": "sum", "src_id": "count"}) ??高度优化的底? C/C++ 向量化?算，??小?的?算??到几秒??。

### 4. 缺乏 Out-of-Core ?算??
**??**：?原本是一?基于 (decision_time, layer_id, scale) 分?的 embarrassingly parallel 任?。正确做法是不??把一天 100 多? Layer 的?文件合并，而是??利用 Polars / DuckDB 的 Out-of-Core ?加?查?，分 Layer ?取、分 Layer 跑?聚?、分 Layer ?入，???存消耗始?能控制在? Layer 的百兆?，且支持全核??并?。

## ?? (Conclusion)
?前的? Pandas ?存堆砌 + 原生 Python 循?策略在面? 1 ???的真?金融?网??已??到架构极限。必?引入向量化（Vectorization）、DuckDB/Polars ?存外流式?理，或使用 SciPy Sparse Matrix 替?嵌套字典?重?。
