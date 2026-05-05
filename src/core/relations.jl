# ============================================================
# DirectRelation：用户-物品直接交互关系
# ============================================================

mutable struct DirectRelation{M<:AbstractMatrix} <: AbstractRelation
    matrix::M
    user_ids::Vector{Int}
    item_ids::Vector{Int}
    relation_type::Symbol  # :rating, :binary, :count

    function DirectRelation(
        matrix::M,
        user_ids::Vector{Int},
        item_ids::Vector{Int},
        relation_type::Symbol
    ) where M<:AbstractMatrix
        size(matrix, 1) == length(user_ids) ||
            throw(DimensionMismatch("矩阵行数必须等于用户数"))
        size(matrix, 2) == length(item_ids) ||
            throw(DimensionMismatch("矩阵列数必须等于物品数"))
        return new{M}(matrix, user_ids, item_ids, relation_type)
    end
end

# 数据访问

function get_user_items(rel::DirectRelation, user_id::Int)
    user_idx = findfirst(x -> x == user_id, rel.user_ids)
    if user_idx === nothing
        return Dict{Int, Float64}()
    end
    row = rel.matrix[user_idx, :]
    indices, values = findnz(row)
    result = Dict{Int, Float64}()
    for (idx, val) in zip(indices, values)
        result[rel.item_ids[idx]] = val
    end
    return result
end

function get_item_users(rel::DirectRelation, item_id::Int)
    item_idx = findfirst(x -> x == item_id, rel.item_ids)
    if item_idx === nothing
        return Dict{Int, Float64}()
    end
    col = rel.matrix[:, item_idx]
    indices, values = findnz(col)
    result = Dict{Int, Float64}()
    for (idx, val) in zip(indices, values)
        result[rel.user_ids[idx]] = val
    end
    return result
end

function has_interacted(rel::DirectRelation, user_id::Int, item_id::Int)
    user_idx = findfirst(x -> x == user_id, rel.user_ids)
    item_idx = findfirst(x -> x == item_id, rel.item_ids)
    if user_idx === nothing || item_idx === nothing
        return false
    end
    return rel.matrix[user_idx, item_idx] != 0
end
