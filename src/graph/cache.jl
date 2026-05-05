# ============================================================
# CachedSimilarityGraph：带近邻缓存的相似度图
# ============================================================

mutable struct CachedSimilarityGraph{M<:SimilarityMetric} <: InferredRelation
    matrix::SparseMatrixCSC{Float64}
    metric::M
    neighbor_cache::Dict{Int, Vector{Tuple{Int, Float64}}}

    function CachedSimilarityGraph(
        matrix::SparseMatrixCSC{Float64},
        metric::M
    ) where M<:SimilarityMetric
        return new{M}(matrix, metric, Dict{Int, Vector{Tuple{Int, Float64}}}())
    end
end

function get_neighbors(graph::CachedSimilarityGraph, node_id::Int)
    if !haskey(graph.neighbor_cache, node_id)
        row = graph.matrix[node_id, :]
        indices, values = findnz(row)
        graph.neighbor_cache[node_id] = collect(zip(indices, values))
    end
    return graph.neighbor_cache[node_id]
end

function invalidate_cache!(graph::CachedSimilarityGraph, node_id::Int)
    delete!(graph.neighbor_cache, node_id)
end

function invalidate_cache!(graph::CachedSimilarityGraph)
    empty!(graph.neighbor_cache)
end
