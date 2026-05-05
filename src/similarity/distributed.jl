# ============================================================
# 分布式相似度计算（占位）
# 正式使用需要加载 Distributed.jl 并添加到 worker
# ============================================================

function _split_indices(total::Int, n_parts::Int)
    part_size = ceil(Int, total / n_parts)
    return [(i):min(i + part_size - 1, total) for i in 1:part_size:total]
end

function _compute_chunk(
    mat::SparseMatrixCSC{Float64},
    chunk_indices::UnitRange{Int},
    ::CosineSimilarity,
    k::Int
)
    mat_norm = _normalize_rows(mat)
    I, J, V = Int[], Int[], Float64[]
    batch_sim = mat_norm[chunk_indices, :] * mat_norm'

    for (offset, row_vec) in enumerate(eachrow(batch_sim))
        global_i = chunk_indices[offset]
        top_k = min(k, length(row_vec))
        top_indices = partialsortperm(row_vec, 1:top_k, rev=true)
        for j in top_indices
            push!(I, global_i); push!(J, j); push!(V, row_vec[j])
        end
    end
    return sparse(I, J, V, size(mat, 1), size(mat, 1))
end

function compute_similarity_distributed(
    metric::CosineSimilarity,
    mat::SparseMatrixCSC{Float64},
    pruning::TopKNeighbors;
    n_workers::Int = nprocs()
)
    n_proc = min(n_workers, nprocs())
    if n_proc < 2
        return compute_similarity(metric, UserItemMatrix(mat), pruning)
    end

    n = size(mat, 1)
    chunks = _split_indices(n, n_proc)
    println("[CoFilter] 分布式计算: $n_proc workers, $n 行")

    # 顺序执行各块（用户需自行配置分布式环境）
    results = SparseMatrixCSC{Float64, Int}[]
    for chunk in chunks
        push!(results, _compute_chunk(mat, chunk, metric, pruning.k))
    end
    return reduce(vcat, results)
end

function compute_similarity_auto(
    metric::CosineSimilarity,
    mat::SparseMatrixCSC{Float64},
    pruning::TopKNeighbors;
    threshold::Int = 10000,
    n_workers::Int = nprocs()
)
    if size(mat, 1) < threshold
        return compute_similarity(metric, UserItemMatrix(mat), pruning)
    else
        return compute_similarity_distributed(metric, mat, pruning; n_workers)
    end
end