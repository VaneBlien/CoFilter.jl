# ============================================================
# 核心抽象类型层级
# ============================================================

# 关系
abstract type AbstractRelation end
abstract type InferredRelation <: AbstractRelation end

# 构建器
abstract type RelationBuilder end

# 相似度度量
abstract type SimilarityMetric end

# 裁剪策略
abstract type PruningStrategy end

# 推荐引擎
abstract type AbstractRecommender end

# 融合策略
abstract type FusionStrategy end
