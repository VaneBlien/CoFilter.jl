# ============================================================
# ItemBasedRecommender：基于物品的协同过滤
# ============================================================

struct ItemBasedRecommender <: AbstractRecommender end

function recommend(
    engine::ItemBasedRecommender,
    user_id::Int,
    direct_relation::DirectRelation,
    item_similarity::InferredRelation,
    n::Int
)
    n <= 0 && return Int[]

    if !(user_id in direct_relation.user_ids)
        return get_popular_items(direct_relation, n)
    end

    user_items = get_user_items(direct_relation, user_id)
    isempty(user_items) && return get_popular_items(direct_relation, n)

    scores = Dict{Int, Float64}()
    for (item_id, rating) in user_items
        # 将物品ID映射为矩阵索引
        item_idx = findfirst(x -> x == item_id, direct_relation.item_ids)
        item_idx === nothing && continue

        neighbors = get_neighbors(item_similarity, item_idx)
        for (neighbor_idx, sim) in neighbors
            # 将矩阵索引映射回物品ID
            neighbor_id = direct_relation.item_ids[neighbor_idx]
            neighbor_id == item_id && continue
            haskey(user_items, neighbor_id) && continue
            scores[neighbor_id] = get(scores, neighbor_id, 0.0) + rating * sim
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