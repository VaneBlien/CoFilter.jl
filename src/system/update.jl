# ============================================================
# 增量更新逻辑
# ============================================================

function update!(sys::RecommendationSystem, new_interactions::Vector{Tuple{Int, Int, Float64}})
    isempty(new_interactions) && return sys

    println("[CoFilter] 开始增量更新，新增 $(length(new_interactions)) 条交互")

    # 1. 记录更新前的矩阵尺寸
    old_n_users = length(sys.direct_relation.user_ids)
    old_n_items = length(sys.direct_relation.item_ids)

    # 2. 更新直接关系矩阵
    affected_users, affected_items = _append_interactions!(sys.direct_relation, new_interactions)

    # 3. 增量更新每个推断关系
    for (key, builder) in sys.builders
        println("[CoFilter] -> 增量更新关系: $key")
        sys.inferred_relations[key] = _incremental_update!(
            sys.inferred_relations[key],
            builder,
            sys.direct_relation,
            affected_users,
            affected_items,
            old_n_users,
            old_n_items
        )
    end

    println("[CoFilter] 增量更新完成")
    return sys
end

# ── 追加交互到 DirectRelation ──
function _append_interactions!(rel::DirectRelation, new_interactions::Vector{Tuple{Int, Int, Float64}})
    affected_users = Set{Int}()
    affected_items = Set{Int}()

    new_user_ids = Int[]
    new_item_ids = Int[]

    for (uid, iid, _) in new_interactions
        push!(affected_users, uid)
        push!(affected_items, iid)
        if !(uid in rel.user_ids) && !(uid in new_user_ids)
            push!(new_user_ids, uid)
        end
        if !(iid in rel.item_ids) && !(iid in new_item_ids)
            push!(new_item_ids, iid)
        end
    end

    # 扩展矩阵
    if !isempty(new_user_ids)
        append!(rel.user_ids, new_user_ids)
        rel.matrix = vcat(rel.matrix, spzeros(length(new_user_ids), size(rel.matrix, 2)))
    end
    if !isempty(new_item_ids)
        append!(rel.item_ids, new_item_ids)
        rel.matrix = hcat(rel.matrix, spzeros(size(rel.matrix, 1), length(new_item_ids)))
    end

    # 填充新评分
    for (uid, iid, rating) in new_interactions
        user_idx = findfirst(x -> x == uid, rel.user_ids)
        item_idx = findfirst(x -> x == iid, rel.item_ids)
        rel.matrix[user_idx, item_idx] = rating
    end

    return affected_users, affected_items
end

# ── 增量更新物品相似度图 ──
function _incremental_update!(
    graph::CachedSimilarityGraph,
    builder::ItemSimilarityBuilder,
    relation::DirectRelation,
    affected_users::Set{Int},
    affected_items::Set{Int},
    old_n_users::Int,
    old_n_items::Int
)
    new_n_items = length(relation.item_ids)

    # 如果矩阵变大了，直接全量重算（简单稳妥）
    if new_n_items != old_n_items
        println("[CoFilter]   矩阵尺寸变化，执行全量重算...")
        wrapper = ItemUserMatrix(relation.matrix)
        sim = compute_similarity(builder.metric, wrapper, builder.pruning)
        return CachedSimilarityGraph(sim, builder.metric)
    end

    # 矩阵大小没变，增量更新受影响的行
    for item_id in affected_items
        item_idx = findfirst(x -> x == item_id, relation.item_ids)
        item_idx === nothing && continue

        invalidate_cache!(graph, item_idx)

        wrapper = ItemUserMatrix(relation.matrix)
        full_sim = compute_similarity(builder.metric, wrapper, builder.pruning)
        graph.matrix[item_idx, :] = full_sim[item_idx, :]
        graph.matrix[:, item_idx] = full_sim[:, item_idx]
    end

    return graph
end

# ── 增量更新用户相似度图 ──
function _incremental_update!(
    graph::CachedSimilarityGraph,
    builder::UserSimilarityBuilder,
    relation::DirectRelation,
    affected_users::Set{Int},
    affected_items::Set{Int},
    old_n_users::Int,
    old_n_items::Int
)
    new_n_users = length(relation.user_ids)

    if new_n_users != old_n_users
        println("[CoFilter]   矩阵尺寸变化，执行全量重算...")
        wrapper = UserItemMatrix(relation.matrix)
        sim = compute_similarity(builder.metric, wrapper, builder.pruning)
        return CachedSimilarityGraph(sim, builder.metric)
    end

    for user_id in affected_users
        user_idx = findfirst(x -> x == user_id, relation.user_ids)
        user_idx === nothing && continue

        invalidate_cache!(graph, user_idx)

        wrapper = UserItemMatrix(relation.matrix)
        full_sim = compute_similarity(builder.metric, wrapper, builder.pruning)
        graph.matrix[user_idx, :] = full_sim[user_idx, :]
        graph.matrix[:, user_idx] = full_sim[:, user_idx]
    end

    return graph
end