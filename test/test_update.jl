@testset "增量更新" begin
    using SparseArrays

    # ==========================================
    # 构建初始数据
    # ==========================================
    n_users, n_items = 10, 15
    mat = sprand(n_users, n_items, 0.2)
    rel = DirectRelation(mat, collect(1:n_users), collect(1:n_items), :rating)

    builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
    sys = RecommendationSystem(rel, Dict(:item_sim => builder), ItemBasedRecommender())
    train!(sys)

    original_size = size(sys.inferred_relations[:item_sim].matrix)
    original_nnz = nnz(sys.inferred_relations[:item_sim].matrix)

    # ==========================================
    # 增量更新：新增评分
    # ==========================================
    @testset "新增评分" begin
        new_interactions = [(1, 3, 5.0), (2, 5, 4.0)]
        update!(sys, new_interactions)

        @test size(sys.inferred_relations[:item_sim].matrix) == original_size
        # 缓存应该被清除
        graph = sys.inferred_relations[:item_sim]
        @test isempty(graph.neighbor_cache)
    end

    # ==========================================
    # 增量更新：新用户
    # ==========================================
    @testset "新用户" begin
        new_interactions = [(99, 1, 3.0), (99, 2, 5.0)]
        update!(sys, new_interactions)

        @test 99 in sys.direct_relation.user_ids
        @test length(sys.direct_relation.user_ids) == n_users + 1
    end

    # ==========================================
    # 增量更新：新物品
    # ==========================================
    @testset "新物品" begin
        new_interactions = [(1, 99, 4.0), (3, 99, 2.0)]
        update!(sys, new_interactions)

        @test 99 in sys.direct_relation.item_ids
    end

    # ==========================================
    # 空更新
    # ==========================================
    @testset "空更新" begin
        before_nnz = nnz(sys.direct_relation.matrix)
        update!(sys, Tuple{Int, Int, Float64}[])
        @test nnz(sys.direct_relation.matrix) == before_nnz
    end

    # ==========================================
    # 更新后可推荐
    # ==========================================
    @testset "更新后推荐" begin
        recs = recommend(sys, 1, 5)
        @test length(recs) <= 5
        @test all(r -> r isa Int, recs)
    end
end