@testset "评估" begin
    using SparseArrays

    mat = sprand(20, 30, 0.1)
    rel = DirectRelation(mat, collect(1:20), collect(1:30), :rating)

    # ==========================================
    # train_test_split
    # ==========================================
    @testset "数据划分" begin
        train_rel, test_pairs = train_test_split(rel, 0.2; random_seed=42)

        @test train_rel isa DirectRelation
        @test size(train_rel.matrix) == size(rel.matrix)
        @test nnz(train_rel.matrix) < nnz(rel.matrix)
        @test length(test_pairs) > 0

        # 校验测试对中的用户和物品都在范围内
        for (uid, iid) in test_pairs
            @test uid in rel.user_ids
            @test iid in rel.item_ids
        end
    end

    # ==========================================
    # 参数校验
    # ==========================================
    @testset "train_test_split 参数校验" begin
        @test_throws ArgumentError train_test_split(rel, 0.0)
        @test_throws ArgumentError train_test_split(rel, 1.0)
        @test_throws ArgumentError train_test_split(rel, -0.1)
    end

    # ==========================================
    # evaluate
    # ==========================================
    @testset "离线评估" begin
        train_rel, test_pairs = train_test_split(rel, 0.2; random_seed=42)
        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(10))
        engine = ItemBasedRecommender()

        metrics = evaluate(() -> begin
            sys = RecommendationSystem(train_rel, Dict(:sim => builder), engine)
            train!(sys)
            return sys
        end, test_pairs, 5)

        @test metrics isa Metrics
        @test 0.0 <= metrics.precision <= 1.0
        @test 0.0 <= metrics.recall <= 1.0
        @test 0.0 <= metrics.ndcg <= 1.0
        @test 0.0 <= metrics.hit_rate <= 1.0
    end

    # ==========================================
    # evaluate 空测试集
    # ==========================================
    @testset "空测试集" begin
        metrics = evaluate(() -> nothing, Tuple{Int, Int}[], 5)
        @test metrics.precision == 0.0
        @test metrics.hit_rate == 0.0
    end

    # ==========================================
    # Metrics 显示
    # ==========================================
    @testset "Metrics 输出" begin
        m = Metrics(0.5, 0.3, 0.4, 0.6)
        buf = IOBuffer()
        show(buf, m)
        s = String(take!(buf))
        @test contains(s, "precision")
        @test contains(s, "recall")
        @test contains(s, "ndcg")
    end

    # ==========================================
    # 交叉验证
    # ==========================================
    @testset "交叉验证" begin
        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        engine = ItemBasedRecommender()

        results = cross_validate(rel, builder, engine, 3, 5; random_seed=42)
        @test length(results) == 3
        for m in results
            @test m isa Metrics
            @test 0.0 <= m.precision <= 1.0
        end
    end
end