# ============================================================
# UserBasedRecommender：基于用户的协同过滤
# ============================================================

struct UserBasedRecommender <: AbstractRecommender end

function recommend(
    engine::UserBasedRecommender,
    user_id::Int,
    direct_relation::DirectRelation,
    user_similarity::InferredRelation,
    n::Int
)
    n <= 0 && return Int[]

    if !(user_id in direct_relation.user_ids)
        return get_popular_items(direct_relation, n)
    end

    user_items = get_user_items(direct_relation, user_id)
    isempty(user_items) && return get_popular_items(direct_relation, n)

    # 用户ID转矩阵索引
    user_idx = findfirst(x -> x == user_id, direct_relation.user_ids)
    user_idx === nothing && return get_popular_items(direct_relation, n)

    similar_users = get_neighbors(user_similarity, user_idx)

    scores = Dict{Int, Float64}()
    for (neighbor_idx, sim) in similar_users
        neighbor_id = direct_relation.user_ids[neighbor_idx]
        neighbor_id == user_id && continue
        neighbor_items = get_user_items(direct_relation, neighbor_id)
        for (item_id, rating) in neighbor_items
            haskey(user_items, item_id) && continue
            scores[item_id] = get(scores, item_id, 0.0) + sim * rating
        end
    end

    if length(scores) < n
        popular = get_popular_items(direct_relation, n)
        for pid in popular
            haskey(user_items, pid) && continue
            haskey(scores, pid) && continue
            scores[pid] = 0.0
            length(scores) >= n && break
        end
    end

    isempty(scores) && return get_popular_items(direct_relation, n)

    sorted = sort(collect(keys(scores)), by = id -> scores[id], rev = true)
    return sorted[1:min(n, length(sorted))]
end