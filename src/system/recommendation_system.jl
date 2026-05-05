# ============================================================
# RecommendationSystem：顶层系统组件
# ============================================================

mutable struct RecommendationSystem
    direct_relation::DirectRelation
    builders::Dict{Symbol, RelationBuilder}
    inferred_relations::Dict{Symbol, InferredRelation}
    recommender::AbstractRecommender

    function RecommendationSystem(
        direct_relation::DirectRelation,
        builders::Dict{Symbol, <:RelationBuilder},
        recommender::AbstractRecommender
    )
        # 转换类型保证 Dict 一致
        typed_builders = Dict{Symbol, RelationBuilder}()
        for (k, v) in builders
            typed_builders[k] = v
        end
        return new(
            direct_relation,
            typed_builders,
            Dict{Symbol, InferredRelation}(),
            recommender
        )
    end
end

# 对外推荐接口
function recommend(sys::RecommendationSystem, user_id::Int, n::Int = 10)
    if sys.recommender isa HybridRecommender
        return recommend(sys.recommender, user_id, sys.direct_relation,
                         sys.inferred_relations, n)
    else
        rel = first(values(sys.inferred_relations))
        return recommend(sys.recommender, user_id, sys.direct_relation, rel, n)
    end
end