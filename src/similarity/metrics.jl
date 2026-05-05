# ============================================================
# 相似度度量类型
# ============================================================

struct CosineSimilarity <: SimilarityMetric end
struct PearsonSimilarity <: SimilarityMetric end
struct JaccardSimilarity <: SimilarityMetric end

struct AdjustedCosineSimilarity <: SimilarityMetric
    damping_factor::Float64
    function AdjustedCosineSimilarity(damping_factor::Float64 = 0.5)
        0 <= damping_factor <= 1 ||
            throw(ArgumentError("damping_factor 必须在 [0, 1] 内"))
        return new(damping_factor)
    end
end
