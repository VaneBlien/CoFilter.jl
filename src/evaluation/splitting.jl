# ============================================================
# 数据集划分
# ============================================================

using Random

function train_test_split(
    relation::DirectRelation,
    test_ratio::Float64 = 0.2;
    random_seed::Int = 42
)
    0.0 < test_ratio < 1.0 || throw(ArgumentError("test_ratio 必须在 (0, 1) 内"))

    # 找出所有用户-物品交互对
    I, J, V = findnz(relation.matrix)
    interactions = collect(zip(I, J, V))

    # 按用户分组
    user_groups = Dict{Int, Vector{Tuple{Int, Int, Float64}}}()
    for (row, col, val) in interactions
        user_id = relation.user_ids[row]
        item_id = relation.item_ids[col]
        pair = (user_id, item_id, val)
        if !haskey(user_groups, user_id)
            user_groups[user_id] = []
        end
        push!(user_groups[user_id], pair)
    end

    # 为每个用户留出 test_ratio 比例的交互
    rng = MersenneTwister(random_seed)
    test_pairs = Tuple{Int, Int}[]

    train_I, train_J, train_V = Int[], Int[], Float64[]
    n_users = length(relation.user_ids)
    n_items = length(relation.item_ids)

    for user_id in relation.user_ids
        group = get(user_groups, user_id, [])
        isempty(group) && continue

        n_test = max(1, round(Int, length(group) * test_ratio))
        shuffled = shuffle(rng, group)
        test_group = shuffled[1:n_test]
        train_group = shuffled[n_test + 1:end]

        for (uid, iid, val) in test_group
            push!(test_pairs, (uid, iid))
        end
        for (uid, iid, val) in train_group
            push!(train_I, findfirst(x -> x == uid, relation.user_ids))
            push!(train_J, findfirst(x -> x == iid, relation.item_ids))
            push!(train_V, val)
        end
    end

    # 构建训练矩阵
    train_matrix = sparse(train_I, train_J, train_V, n_users, n_items)
    train_relation = DirectRelation(
        train_matrix,
        relation.user_ids,
        relation.item_ids,
        relation.relation_type
    )

    return train_relation, test_pairs
end