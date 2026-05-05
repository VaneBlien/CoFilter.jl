# ============================================================
# SimilarityGraph：推断关系的基础实现
# ============================================================

struct SimilarityGraph{M<:SimilarityMetric} <: InferredRelation
    matrix::SparseMatrixCSC{Float64}
    metric::M
end

function get_neighbors(graph::SimilarityGraph, node_id::Int)
    row = graph.matrix[node_id, :]
    indices, values = findnz(row)
    return collect(zip(indices, values))
end

function get_similarity(graph::SimilarityGraph, a::Int, b::Int)
    return graph.matrix[a, b]
end
