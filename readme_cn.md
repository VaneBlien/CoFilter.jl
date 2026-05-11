
# CoFilter.jl

**CoFilter** 是一个纯 Julia 实现的协同过滤推荐算法模块，基于**多重分派**和**关系模块化**架构设计。

## 特性

- **关系抽象**：用户-物品交互、相似度图统一为“关系”类型
- **算法即类型**：每种相似度度量和推荐策略都是具体类型
- **多重分派驱动**：零运行时开销的策略选择
- **高性能**：直接操作稀疏矩阵，利用 BLAS3 矩阵乘法加速
- **增量更新**：支持新交互数据的热更新，无需全量重训练
- **离线评估**：内置 Precision、Recall、NDCG、Hit Rate 全套指标

## 架构

```
数据层 (Relations) → 计算层 (Builders) → 策略层 (Recommenders) → Top-N 推荐
```

模块采用清晰的分层结构，将关注点分离并最大化可扩展性：

```
src/
├── CoFilter.jl                  # 模块入口
├── core/                        # 核心类型与抽象
│   ├── interfaces.jl
│   ├── types.jl
│   └── relations.jl
├── similarity/                  # 相似度计算
│   ├── metrics.jl               #   Cosine, Pearson, Jaccard, AdjustedCosine
│   ├── pruning.jl               #   TopK, MinSimilarityThreshold 剪枝策略
│   ├── computation.jl           #   核心计算内核
│   ├── builders.jl              #   构建器模式组装计算管线
│   └── distributed.jl           #   多工作进程分布式计算
├── recommenders/                # 推荐引擎
│   ├── base.jl
│   ├── user_based.jl
│   ├── item_based.jl
│   ├── fusion.jl                #   WeightedSum & RoundRobin 融合策略
│   └── hybrid.jl
├── graph/                       # 相似度图表示
│   ├── similarity_graph.jl
│   └── cache.jl                 #   透明近邻缓存
├── system/                      # 系统组装与生命周期
│   ├── recommendation_system.jl
│   ├── training.jl              #   全量训练
│   └── update.jl                #   增量在线更新
├── evaluation/                  # 离线评估套件
│   ├── splitting.jl             #   训练/测试集划分
│   ├── metrics.jl               #   Precision, Recall, NDCG, Hit Rate
│   └── cross_validation.jl
└── utils/                       # 工具模块
    ├── sparse_utils.jl
    ├── cold_start.jl            #   新用户/物品冷启动兜底
    └── validation.jl
```

这种模块化设计允许你独立地交换相似度度量、剪枝策略和推荐引擎，或将它们组合成混合推荐系统。

## 快速开始

### 安装

该包尚未注册到 Julia 注册表，可通过 GitHub 直接安装：

```julia
using Pkg
Pkg.add(url="https://github.com/VaneBlien/CoFilter.jl")
```

或克隆仓库后通过本地路径加载：

```julia
using Pkg
Pkg.develop(path="path/to/CoFilter.jl")
```

### 基础使用

```julia
using CoFilter, SparseArrays

# 1. 准备数据: 100 位用户, 200 个物品, 5% 稠密度
n_users, n_items = 100, 200
matrix = sprand(n_users, n_items, 0.05)
direct_rel = DirectRelation(matrix, collect(1:n_users), collect(1:n_items), :rating)

# 2. 配置基于物品的协同过滤系统
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
sys = RecommendationSystem(direct_rel, Dict(:item_sim => builder), ItemBasedRecommender())

# 3. 训练
train!(sys)

# 4. 为用户 1 推荐 Top-10 物品
top10 = recommend(sys, 1, 10)
println("为用户 1 推荐: $top10")
```

### 离线评估

```julia
# 划分训练集与测试集
train_rel, test_pairs = train_test_split(direct_rel, 0.2)

# 使用工厂闭包进行评估
metrics = evaluate(() -> begin
    sys = RecommendationSystem(train_rel, Dict(:item_sim => builder), engine)
    train!(sys)
    return sys
end, test_pairs, 10)

println("Precision@10: $(metrics.precision)")
println("Recall@10:    $(metrics.recall)")
println("NDCG@10:      $(metrics.ndcg)")
```

### MovieLens 示例

内置 MovieLens 100K 数据集上的完整端到端示例：

```julia
include("examples/movielens_demo.jl")
```

## 支持的算法

| 类别 | 算法 | 类型 |
|------|------|------|
| 协同过滤 | 基于用户的协同过滤 | `UserBasedRecommender` |
| 协同过滤 | 基于物品的协同过滤 | `ItemBasedRecommender` |
| 混合推荐 | 加权融合 | `HybridRecommender` + `WeightedSum` |
| 混合推荐 | 轮询融合 | `HybridRecommender` + `RoundRobin` |

## 相似度度量

- `CosineSimilarity` — 余弦相似度
- `PearsonSimilarity` — 皮尔逊相关系数
- `JaccardSimilarity` — Jaccard 系数
- `AdjustedCosineSimilarity` — 带阻尼的调整余弦相似度

## 剪枝策略

- `TopKNeighbors(k)` — 仅保留 K 个最近邻
- `MinSimilarityThreshold(t)` — 保留相似度 ≥ t 的近邻

## 评估指标

- Precision@K
- Recall@K
- NDCG@K
- Hit Rate@K

## 测试

运行完整测试套件（覆盖所有模块的 225 项测试）：

```julia
using Pkg
Pkg.test("CoFilter")
```

## 路线图

参见 [ROADMAP.md](ROADMAP.md) 了解计划中的算法扩展、性能优化和工程改进。

## 许可

本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。
