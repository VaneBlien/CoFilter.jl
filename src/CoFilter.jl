module CoFilter

# 外部依赖
using SparseArrays
using LinearAlgebra
using Statistics
using Distributed
using Serialization

# ============================================================
# 导出列表
# ============================================================

# 核心类型
export AbstractRelation, DirectRelation, InferredRelation
export SimilarityGraph, CachedSimilarityGraph

# 相似度
export SimilarityMetric
export CosineSimilarity, PearsonSimilarity, JaccardSimilarity
export AdjustedCosineSimilarity
export UserItemMatrix, ItemUserMatrix

# 裁剪策略
export PruningStrategy, TopKNeighbors, MinSimilarityThreshold

# 构建器
export RelationBuilder
export UserSimilarityBuilder, ItemSimilarityBuilder

# 推荐引擎
export AbstractRecommender
export UserBasedRecommender, ItemBasedRecommender
export HybridRecommender
export FusionStrategy, WeightedSum, RoundRobin

# 系统
export RecommendationSystem
export train!, update!, recommend

# 评估
export Metrics
export evaluate, cross_validate, train_test_split

# 核心函数
export build, compute_similarity
export get_user_items, get_item_users
export get_neighbors, get_similarity
export compute_similarity_distributed, compute_similarity_auto

# ============================================================
# 模块加载顺序
# ============================================================

# 1. 核心类型与接口
include("core/interfaces.jl")
include("core/types.jl")
include("core/relations.jl")

# 2. 相似度计算层
include("similarity/metrics.jl")
include("similarity/pruning.jl")
include("similarity/computation.jl")
include("similarity/builders.jl")
include("similarity/distributed.jl")

# 3. 相似度图
include("graph/similarity_graph.jl")
include("graph/cache.jl")

# 4. 推荐引擎层
include("recommenders/base.jl")
include("recommenders/user_based.jl")
include("recommenders/item_based.jl")
include("recommenders/fusion.jl")
include("recommenders/hybrid.jl")

# 5. 系统层
include("system/recommendation_system.jl")
include("system/training.jl")
include("system/update.jl")

# 6. 评估层
include("evaluation/splitting.jl")
include("evaluation/metrics.jl")
include("evaluation/cross_validation.jl")

# 7. 工具层
include("utils/sparse_utils.jl")
include("utils/cold_start.jl")
include("utils/validation.jl")

end # module CoFilter
