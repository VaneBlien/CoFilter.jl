# ============================================================
# 交叉验证
# ============================================================

function cross_validate(
    relation::DirectRelation,
    builder::RelationBuilder,
    engine::AbstractRecommender,
    n_folds::Int = 5,
    k::Int = 10;
    random_seed::Int = 42
)::Vector{Metrics}
    n_folds >= 2 || throw(ArgumentError("n_folds 必须 >= 2"))

    # 获取所有交互对
    I, J, V = findnz(relation.matrix)
    all_pairs = Tuple{Int, Int}[]
    for (row, col, _) in zip(I, J, V)
        push!(all_pairs, (relation.user_ids[row], relation.item_ids[col]))
    end

    n_total = length(all_pairs)
    fold_size = ceil(Int, n_total / n_folds)

    rng = MersenneTwister(random_seed)
    shuffled = shuffle(rng, all_pairs)

    results = Metrics[]

    for fold in 1:n_folds
        test_start = (fold - 1) * fold_size + 1
        test_end = min(fold * fold_size, n_total)
        test_start > n_total && break

        test_pairs = shuffled[test_start:test_end]
        train_pairs = vcat(shuffled[1:test_start-1], shuffled[test_end+1:end])

        # 构建训练矩阵
        n_users = length(relation.user_ids)
        n_items = length(relation.item_ids)
        train_I, train_J, train_V = Int[], Int[], Float64[]
        for (uid, iid) in train_pairs
            row = findfirst(x -> x == uid, relation.user_ids)
            col = findfirst(x -> x == iid, relation.item_ids)
            push!(train_I, row)
            push!(train_J, col)
            push!(train_V, relation.matrix[row, col])
        end

        train_matrix = sparse(train_I, train_J, train_V, n_users, n_items)
        train_rel = DirectRelation(
            train_matrix,
            relation.user_ids,
            relation.item_ids,
            relation.relation_type
        )

        # 构建系统并训练
        sys = RecommendationSystem(train_rel, Dict(:sim => builder), engine)
        train!(sys)

        # 评估单折
        fold_metrics = evaluate(() -> begin
            fold_sys = RecommendationSystem(train_rel, Dict(:sim => builder), engine)
            train!(fold_sys)
            return fold_sys
        end, test_pairs, k)

        push!(results, fold_metrics)
    end

    return results
end