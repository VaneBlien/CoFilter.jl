# ============================================================
# 构建器实现
# ============================================================

struct UserSimilarityBuilder{M<:SimilarityMetric, P<:PruningStrategy} <: RelationBuilder
    metric::M
    pruning::P

    function UserSimilarityBuilder(
        metric::SimilarityMetric = CosineSimilarity(),
        pruning::PruningStrategy = TopKNeighbors(50)
    )
        return new{typeof(metric), typeof(pruning)}(metric, pruning)
    end
end

struct ItemSimilarityBuilder{M<:SimilarityMetric, P<:PruningStrategy} <: RelationBuilder
    metric::M
    pruning::P

    function ItemSimilarityBuilder(
        metric::SimilarityMetric = CosineSimilarity(),
        pruning::PruningStrategy = TopKNeighbors(50)
    )
        return new{typeof(metric), typeof(pruning)}(metric, pruning)
    end
end
