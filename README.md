# CoFilter.jl

基于多重分派的关系模块化协同过滤推荐算法框架。

## 架构

```
数据层 (Relations) -> 计算层 (Builders) -> 策略层 (Recommenders) -> 推荐结果
```

## 快速开始

```julia
using CoFilter

# 准备数据
matrix = sprand(1000, 500, 0.05)
direct_rel = DirectRelation(matrix, 1:1000, 1:500, :rating)

# 配置系统
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(50))
sys = RecommendationSystem(direct_rel, Dict(:item_sim => builder), ItemBasedRecommender())

# 训练并推荐
train!(sys)
recommend(sys, 42, 10)
```

## 许可

MIT License
