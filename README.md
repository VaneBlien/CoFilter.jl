
# CoFilter.jl

**CoFilter** 是一个纯 Julia 实现的协同过滤推荐算法模块，基于**多重分派**和**关系模块化**架构设计。

## 特性

- 关系抽象：用户-物品交互、相似度图统一为"关系"类型
- 算法即类型：每种相似度度量和推荐策略都是具体类型
- 多重分派驱动：零运行时开销的策略选择
- 高性能：直接操作稀疏矩阵，利用 BLAS3 矩阵乘法加速
- 增量更新：支持新交互数据的热更新
- 离线评估：Precision、Recall、NDCG、Hit Rate 全套指标

## 架构

```
数据层 (Relations) → 计算层 (Builders) → 策略层 (Recommenders) → 推荐结果
```

## 快速开始

### 安装

由于包尚未注册，请通过本地路径安装：

```julia
using Pkg
Pkg.develop(path="path/to/CoFilter.jl")
```

### 基础使用

```julia
using CoFilter, SparseArrays

# 1. 准备数据
n_users, n_items = 100, 200
matrix = sprand(n_users, n_items, 0.05)
direct_rel = DirectRelation(matrix, collect(1:n_users), collect(1:n_items), :rating)

# 2. 配置系统（基于物品的协同过滤）
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
sys = RecommendationSystem(direct_rel, Dict(:item_sim => builder), ItemBasedRecommender())

# 3. 训练
train!(sys)

# 4. 推荐
top10 = recommend(sys, 1, 10)
println("为用户 1 推荐: $top10")
```

### 离线评估

```julia
# 划分训练/测试集
train_rel, test_pairs = train_test_split(direct_rel, 0.2)

# 评估
metrics = evaluate(() -> begin
    sys = RecommendationSystem(train_rel, Dict(:item_sim => builder), engine)
    train!(sys)
    return sys
end, test_pairs, 10)
```

### MovieLens 示例

```julia
include("examples/movielens_demo.jl")
```

## 支持的算法

| 类别 | 算法 | 类型 |
|------|------|------|
| 协同过滤 | User-Based CF | `UserBasedRecommender` |
| 协同过滤 | Item-Based CF | `ItemBasedRecommender` |
| 混合 | 加权融合 | `HybridRecommender` + `WeightedSum` |
| 混合 | 轮询融合 | `HybridRecommender` + `RoundRobin` |

## 相似度度量

- `CosineSimilarity` — 余弦相似度
- `PearsonSimilarity` — 皮尔逊相关系数
- `JaccardSimilarity` — Jaccard 系数
- `AdjustedCosineSimilarity` — 带阻尼的调整余弦相似度

## 裁剪策略

- `TopKNeighbors(k)` — 保留 K 个最近邻
- `MinSimilarityThreshold(t)` — 保留相似度 ≥ t 的近邻

## 评估指标

- Precision@K
- Recall@K
- NDCG@K
- Hit Rate@K

## 项目结构

```
src/
├── CoFilter.jl              # 模块入口
├── core/                    # 核心类型与抽象
│   ├── interfaces.jl
│   ├── types.jl
│   └── relations.jl
├── similarity/              # 相似度计算
│   ├── metrics.jl
│   ├── pruning.jl
│   ├── computation.jl
│   ├── builders.jl
│   └── distributed.jl
├── recommenders/            # 推荐引擎
│   ├── base.jl
│   ├── user_based.jl
│   ├── item_based.jl
│   ├── fusion.jl
│   └── hybrid.jl
├── graph/                   # 相似度图
│   ├── similarity_graph.jl
│   └── cache.jl
├── system/                  # 系统组装
│   ├── recommendation_system.jl
│   ├── training.jl
│   └── update.jl
├── evaluation/              # 评估
│   ├── splitting.jl
│   ├── metrics.jl
│   └── cross_validation.jl
└── utils/                   # 工具
    ├── sparse_utils.jl
    ├── cold_start.jl
    └── validation.jl
```

## 测试

```julia
using Pkg
Pkg.test("CoFilter")
```

## 许可

MIT License




