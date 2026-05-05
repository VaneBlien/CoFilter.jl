# ============================================================
# 训练逻辑
# ============================================================

function train!(sys::RecommendationSystem)
    println("[CoFilter] 开始训练...")
    for (key, builder) in sys.builders
        println("[CoFilter] -> 构建关系: $key")
        sys.inferred_relations[key] = build(builder, sys.direct_relation)
    end
    println("[CoFilter] 训练完成")
    return sys
end

function build(builder::ItemSimilarityBuilder, relation::DirectRelation)
    wrapper = ItemUserMatrix(relation.matrix)
    sim = compute_similarity(builder.metric, wrapper, builder.pruning)
    return CachedSimilarityGraph(sim, builder.metric)
end

function build(builder::UserSimilarityBuilder, relation::DirectRelation)
    wrapper = UserItemMatrix(relation.matrix)
    sim = compute_similarity(builder.metric, wrapper, builder.pruning)
    return CachedSimilarityGraph(sim, builder.metric)
end
