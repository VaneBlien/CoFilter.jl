# ============================================================
# 高性能相似度计算内核
# ============================================================

# ── 矩阵视角包装（不复制数据） ──
struct UserItemMatrix{T}
    data::T
end

struct ItemUserMatrix{T}
    data::T
end

# ── 分派入口 ──
function compute_similarity(
    metric::SimilarityMetric,
    wrapper::UserItemMatrix,
    pruning::PruningStrategy
)
    return _compute_impl(metric, wrapper.data, pruning)
end

function compute_similarity(
    metric::SimilarityMetric,
    wrapper::ItemUserMatrix,
    pruning::PruningStrategy
)
    return _compute_impl(metric, SparseMatrixCSC(transpose(wrapper.data)), pruning)
end

# ── 预处理：归一化 ──
function _normalize_rows(mat::SparseMatrixCSC{Float64})
    norms = sqrt.(sum(abs2, mat; dims=2))
    norms[norms .== 0] .= 1.0
    return mat ./ norms
end

# ── 实现：Cosine + TopK ──
function _compute_impl(
    ::CosineSimilarity,
    mat::SparseMatrixCSC{Float64},
    pruning::TopKNeighbors
)
    mat_norm = _normalize_rows(mat)
    n = size(mat_norm, 1)
    I, J, V = Int[], Int[], Float64[]
    batch_size = 1000

    for i_start in 1:batch_size:n
        i_end = min(i_start + batch_size - 1, n)
        batch_sim = mat_norm[i_start:i_end, :] * mat_norm'

        for (offset, row_vec) in enumerate(eachrow(batch_sim))
            global_i = i_start + offset - 1
            k = min(pruning.k, length(row_vec))
            top_indices = partialsortperm(row_vec, 1:k, rev=true)
            for j in top_indices
                push!(I, global_i)
                push!(J, j)
                push!(V, row_vec[j])
            end
        end
    end

    return sparse(I, J, V, n, n)
end

# ── 实现：Cosine + Threshold ──
function _compute_impl(
    ::CosineSimilarity,
    mat::SparseMatrixCSC{Float64},
    pruning::MinSimilarityThreshold
)
    mat_norm = _normalize_rows(mat)
    sim = mat_norm * mat_norm'
    sim[sim .< pruning.threshold] .= 0.0
    return sparse(sim)
end

# ── 实现：Pearson + TopK ──
function _compute_impl(
    ::PearsonSimilarity,
    mat::SparseMatrixCSC{Float64},
    pruning::TopKNeighbors
)
    row_means = vec(mean(mat, dims=2))
    mat_centered = mat .- row_means
    mat_centered[mat .== 0] .= 0.0
    return _compute_impl(CosineSimilarity(), mat_centered, pruning)
end
