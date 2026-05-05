# ============================================================
# ============================================================
# 评估指标
# ============================================================

struct Metrics
    precision::Float64
    recall::Float64
    ndcg::Float64
    hit_rate::Float64
end

function Base.show(io::IO, m::Metrics)
    println(io, "Metrics(")
    println(io, "  precision = ", round(m.precision, digits=4))
    println(io, "  recall    = ", round(m.recall, digits=4))
    println(io, "  ndcg      = ", round(m.ndcg, digits=4))
    println(io, "  hit_rate  = ", round(m.hit_rate, digits=4))
    print(io, ")")
end

function evaluate(
    sys_builder::Function,
    test_pairs::Vector{Tuple{Int, Int}},
    k::Int = 10
)::Metrics
    n = length(test_pairs)
    n == 0 && return Metrics(0.0, 0.0, 0.0, 0.0)

    hits = 0
    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0

    for (user_id, true_item) in test_pairs
        sys = sys_builder()
        recommendations = recommend(sys, user_id, k)

        if true_item in recommendations
            hits += 1
            rank = findfirst(x -> x == true_item, recommendations)
            total_precision += 1.0 / k
            total_recall += 1.0  # 留一法中命中即召回=1
            total_ndcg += 1.0 / log2(rank + 1)
        end
    end

    return Metrics(
        total_precision / n,
        hits / n,
        total_ndcg / n,
        hits / n
    )
end