@testset "推荐系统" begin
    using SparseArrays

    mat = sprand(10, 20, 0.3)
    rel = DirectRelation(mat, collect(1:10), collect(1:20), :rating)

    # ==========================================
    # 系统构建 + 训练
    # ==========================================
    @testset "构建与训练" begin
        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        sys = RecommendationSystem(rel, Dict(:item_sim => builder), ItemBasedRecommender())

        @test isempty(sys.inferred_relations)
        train!(sys)
        @test length(sys.inferred_relations) == 1
        @test :item_sim in keys(sys.inferred_relations)
    end

    # ==========================================
    # 系统推荐
    # ==========================================
    @testset "系统推荐" begin
        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        sys = RecommendationSystem(rel, Dict(:item_sim => builder), ItemBasedRecommender())
        train!(sys)

        recs = recommend(sys, 1, 5)
        @test length(recs) <= 5
        @test all(r -> r isa Int, recs)
    end

    # ==========================================
    # 多构建器
    # ==========================================
    @testset "多构建器" begin
        item_b = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        user_b = UserSimilarityBuilder(PearsonSimilarity(), TopKNeighbors(5))
        sys = RecommendationSystem(rel,
            Dict(:item_sim => item_b, :user_sim => user_b),
            ItemBasedRecommender())
        train!(sys)
        @test length(sys.inferred_relations) == 2
        @test size(sys.inferred_relations[:item_sim].matrix) == (20, 20)
        @test size(sys.inferred_relations[:user_sim].matrix) == (10, 10)
    end

    # ==========================================
    # HybridRecommender 在系统中
    # ==========================================
    @testset "HybridRecommender 系统" begin
        item_b = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        user_b = UserSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        hybrid = HybridRecommender(
            [ItemBasedRecommender(), UserBasedRecommender()],
            WeightedSum([0.6, 0.4])
        )
        sys = RecommendationSystem(rel,
            Dict(:item_sim => item_b, :user_sim => user_b),
            hybrid)
        train!(sys)

        recs = recommend(sys, 1, 5)
        @test length(recs) <= 5
        @test all(r -> r isa Int, recs)
    end
end