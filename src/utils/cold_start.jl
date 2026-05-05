# ============================================================
# 冷启动兜底策略
# ============================================================

function get_popular_items(relation::DirectRelation, n::Int)
    n <= 0 && return Int[]
    pop = vec(sum(relation.matrix .!= 0, dims=1))
    if all(pop .== 0)
        # 完全没有任何交互，返回所有物品 ID
        return relation.item_ids[1:min(n, length(relation.item_ids))]
    end
    idx = sortperm(pop, rev = true)
    top = idx[1:min(n, length(idx))]
    return [relation.item_ids[i] for i in top]
end