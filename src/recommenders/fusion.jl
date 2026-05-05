# ============================================================
# 融合策略
# ============================================================

struct WeightedSum <: FusionStrategy
    weights::Vector{Float64}
end

struct RoundRobin <: FusionStrategy end

function fuse(strategy::WeightedSum, lists::Vector{Int}...)
    scores = Dict{Int, Float64}()
    for (i, lst) in enumerate(lists)
        w = strategy.weights[i]
        for (rank, item_id) in enumerate(lst)
            scores[item_id] = get(scores, item_id, 0.0) + w / (rank + 1)
        end
    end
    sorted = sort(collect(scores), by = x -> x[2], rev = true)
    return first.(sorted)
end

function fuse(::RoundRobin, lists::Vector{Int}...)
    result = Int[]
    max_len = maximum(length.(lists))
    for i in 1:max_len
        for lst in lists
            if i <= length(lst) && !(lst[i] in result)
                push!(result, lst[i])
            end
        end
    end
    return result
end