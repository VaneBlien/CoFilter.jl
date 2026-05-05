# ============================================================
# 裁剪策略
# ============================================================

struct TopKNeighbors <: PruningStrategy
    k::Int
    function TopKNeighbors(k::Int)
        k > 0 || throw(ArgumentError("k 必须为正整数"))
        return new(k)
    end
end

struct MinSimilarityThreshold <: PruningStrategy
    threshold::Float64
    function MinSimilarityThreshold(threshold::Float64)
        0 <= threshold <= 1 ||
            throw(ArgumentError("threshold 必须在 [0, 1] 内"))
        return new(threshold)
    end
end
