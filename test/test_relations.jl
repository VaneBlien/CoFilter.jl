@testset "DirectRelation" begin
    using SparseArrays

    n_users, n_items = 10, 20
    mat = sprand(n_users, n_items, 0.3)
    user_ids = collect(1:n_users)
    item_ids = collect(1:n_items)

    rel = DirectRelation(mat, user_ids, item_ids, :rating)
    @test rel.matrix === mat
    @test rel.user_ids == user_ids
    @test rel.item_ids == item_ids
    @test rel.relation_type == :rating
    @test size(rel.matrix) == (n_users, n_items)

    rel_bin = DirectRelation(mat, user_ids, item_ids, :binary)
    @test rel_bin.relation_type == :binary

    # 维度校验：用 Vector{Int} 但长度不对
    @test_throws DimensionMismatch DirectRelation(mat, collect(1:9), item_ids, :rating)
    @test_throws DimensionMismatch DirectRelation(mat, user_ids, collect(1:19), :rating)

    # get_user_items
    dense = zeros(5, 5)
    dense[1, 2] = 4.0; dense[1, 4] = 5.0
    dense[3, 1] = 2.0
    mat2 = sparse(dense)
    rel2 = DirectRelation(mat2, [101, 102, 103, 104, 105], [201, 202, 203, 204, 205], :rating)

    items1 = get_user_items(rel2, 101)
    @test length(items1) == 2
    @test items1[202] == 4.0
    @test items1[204] == 5.0

    items3 = get_user_items(rel2, 103)
    @test length(items3) == 1
    @test items3[201] == 2.0

    items_empty = get_user_items(rel2, 102)
    @test isempty(items_empty)

    items_none = get_user_items(rel2, 999)
    @test isempty(items_none)

    # get_item_users
    users_of_202 = get_item_users(rel2, 202)
    @test length(users_of_202) == 1
    @test users_of_202[101] == 4.0

    users_of_203 = get_item_users(rel2, 203)
    @test isempty(users_of_203)

    users_none = get_item_users(rel2, 999)
    @test isempty(users_none)

    # has_interacted
    @test CoFilter.has_interacted(rel2, 101, 202) == true
    @test CoFilter.has_interacted(rel2, 101, 204) == true
    @test CoFilter.has_interacted(rel2, 101, 201) == false
    @test CoFilter.has_interacted(rel2, 102, 202) == false
    @test CoFilter.has_interacted(rel2, 999, 202) == false
    @test CoFilter.has_interacted(rel2, 101, 999) == false
end