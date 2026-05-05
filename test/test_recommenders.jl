@testset "推荐引擎" begin
    using SparseArrays

    # 构造确定性矩阵
    dense = zeros(5, 6)
    dense[1, 1] = 5.0; dense[1, 2] = 3.0
    dense[2, 1] = 4.0; dense[2, 3] = 2.0
    dense[3, 2] = 1.0; dense[3, 4] = 5.0
    dense[4, 5] = 3.0; dense[4, 6] = 4.0
    dense[5, 1] = 2.0; dense[5, 3] = 5.0
    mat = sparse(dense)
    rel = DirectRelation(mat, collect(1:5), collect(1:6), :rating)

    # ==========================================
    # ItemBasedRecommender
    # ==========================================
    @testset "ItemBasedRecommender" begin
        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        graph = build(builder, rel)
        engine = ItemBasedRecommender()

        recs = recommend(engine, 1, rel, graph, 3)
        @test length(recs) <= 3
        @test all(r -> r isa Int, recs)
        # 推荐的物品不能是用户已交互的
        user_items = Set(keys(get_user_items(rel, 1)))
        @test all(r -> !(r in user_items), recs)
    end

    # ==========================================
    # UserBasedRecommender
    # ==========================================
    @testset "UserBasedRecommender" begin
        builder = UserSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        graph = build(builder, rel)
        engine = UserBasedRecommender()

        recs = recommend(engine, 1, rel, graph, 3)
        @test length(recs) <= 3
        @test all(r -> r isa Int, recs)
        user_items = Set(keys(get_user_items(rel, 1)))
        @test all(r -> !(r in user_items), recs)
    end

    # ==========================================
    # 冷启动：用户无交互
    # ==========================================
    @testset "冷启动兜底" begin
        # 用户2 只有 item1 和 item3，构造空用户
        empty_dense = zeros(2, 3)
        empty_dense[1, 1] = 3.0
        empty_mat = sparse(empty_dense)
        empty_rel = DirectRelation(empty_mat, [1, 2], [1, 2, 3], :rating)

        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        graph = build(builder, empty_rel)
        engine = ItemBasedRecommender()

        # 用户2 无交互
        recs = recommend(engine, 2, empty_rel, graph, 2)
        @test length(recs) <= 2
        @test all(r -> r isa Int, recs)
    end

    # ==========================================
    # 边界：n=0
    # ==========================================
    @testset "n=0 返回空数组" begin
        builder = ItemSimilarityBuilder()
        graph = build(builder, rel)
        engine = ItemBasedRecommender()

        recs = recommend(engine, 1, rel, graph, 0)
        @test isempty(recs)
    end

    # ==========================================
    # 边界：不存在用户
    # ==========================================
    @testset "不存在用户" begin
        builder = ItemSimilarityBuilder()
        graph = build(builder, rel)
        engine = ItemBasedRecommender()

        recs = recommend(engine, 999, rel, graph, 5)
        @test all(r -> r isa Int, recs)
    end

    # ==========================================
    # HybridRecommender
    # ==========================================
        @testset "HybridRecommender" begin
        item_builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        user_builder = UserSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))

        item_graph = build(item_builder, rel)
        user_graph = build(user_builder, rel)

        hybrid = HybridRecommender(
            [ItemBasedRecommender(), UserBasedRecommender()],
            WeightedSum([0.5, 0.5])
        )

        inferred = Dict{Symbol, InferredRelation}(
            :item_sim => item_graph,
            :user_sim => user_graph
        )

        recs = recommend(hybrid, 1, rel, inferred, 5)
        @test length(recs) <= 5
        @test all(r -> r isa Int, recs)

        hybrid_rr = HybridRecommender(
            [ItemBasedRecommender(), UserBasedRecommender()],
            RoundRobin()
        )
        recs_rr = recommend(hybrid_rr, 1, rel, inferred, 5)
        @test length(recs_rr) <= 5
    end
end