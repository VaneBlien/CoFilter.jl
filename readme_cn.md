
# CoFilter.jl

[English](README.md)

**纯 Julia 协同过滤，围绕稀疏矩阵和多重分派构建。**

## 做了什么

给定用户-物品交互矩阵，CoFilter 构建相似度图并输出 Top-N 推荐。支持 User-Based CF、Item-Based CF、Hybrid 融合——共享同一套稀疏矩阵基础设施，无 Python 依赖。

## 设计选择

- **直接操作 `SparseMatrixCSC`**。相似度计算直接对原生稀疏格式调用 BLAS3，不转稠密，不复制。见 `src/similarity/computation.jl`。
- **算法即具体类型**。`UserBasedRecommender`、`ItemBasedRecommender`、`HybridRecommender` 是不同类型，分派在编译期选择正确路径——没有运行时 `if` 链。
- **Builder 分离度量与策略**。一个 `Builder` 把相似度度量（Cosine、Pearson、Jaccard、AdjustedCosine）和剪枝策略（TopK、阈值）组合在一起。任何度量都可以和任何剪枝组合，不修改推荐逻辑。见 `src/similarity/builders.jl`。
- **增量更新只触及受影响行**。新交互不触发全量重算。见 `src/system/update.jl`。
- **近邻缓存**。`SimilarityGraph` 有可选的 `CachedSimilarityGraph` 变体，首次访问后存储近邻列表。见 `src/graph/cache.jl`。

## 快速开始

```julia
using CoFilter, SparseArrays

# 100 位用户, 200 个物品, 5% 稠密度
matrix = sprand(100, 200, 0.05)
direct_rel = DirectRelation(matrix, collect(1:100), collect(1:200), :rating)

# 基于物品的 CF，余弦相似度，保留 30 个近邻
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
sys = RecommendationSystem(direct_rel, Dict(:item_sim => builder), ItemBasedRecommender())

train!(sys)
top10 = recommend(sys, 1, 10)
```

## 支持的算法

| 类型 | 构造器 |
|------|--------|
| 基于用户的 CF | `UserBasedRecommender()` |
| 基于物品的 CF | `ItemBasedRecommender()` |
| 加权混合 | `HybridRecommender(engines, WeightedSum(weights))` |
| 轮询混合 | `HybridRecommender(engines, RoundRobin())` |

## 相似度度量

`CosineSimilarity()`、`PearsonSimilarity()`、`JaccardSimilarity()`、`AdjustedCosineSimilarity(damping_factor)`

## 剪枝策略

`TopKNeighbors(k)`、`MinSimilarityThreshold(t)`

## 评估指标

Precision@K、Recall@K、NDCG@K、Hit Rate@K —— 见 `src/evaluation/metrics.jl`

## 测试

```julia
using Pkg
Pkg.test("CoFilter")
```

覆盖所有模块，225 项测试。

## 源码结构

```
src/
├── CoFilter.jl
├── core/           # AbstractRelation, DirectRelation, 接口
├── similarity/     # 度量, 剪枝, 计算, 构建器, 分布式
├── recommenders/   # 基于用户, 基于物品, 混合, 融合策略
├── graph/          # SimilarityGraph, CachedSimilarityGraph
├── system/         # RecommendationSystem, 训练, 增量更新
├── evaluation/     # 训练/测试划分, 指标, 交叉验证
└── utils/          # 稀疏工具, 冷启动兜底
```

## 许可

MIT
