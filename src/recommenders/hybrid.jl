# ============================================================
# HybridRecommender：混合推荐引擎
# ============================================================

struct HybridRecommender <: AbstractRecommender
    engines::Vector{AbstractRecommender}
    fusion::FusionStrategy
end

function recommend(
    engine::HybridRecommender,
    user_id::Int,
    direct_relation::DirectRelation,
    inferred_relations::Dict{Symbol, InferredRelation},
    n::Int
)
    lists = Vector{Int}[]
    for sub_engine in engine.engines
        key = _engine_key(sub_engine)
        rel = inferred_relations[key]
        push!(lists, recommend(sub_engine, user_id, direct_relation, rel, n))
    end
    fused = fuse(engine.fusion, lists...)
    return fused[1:min(n, length(fused))]
end

_engine_key(::UserBasedRecommender) = :user_sim
_engine_key(::ItemBasedRecommender) = :item_sim