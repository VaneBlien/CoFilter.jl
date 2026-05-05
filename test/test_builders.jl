@testset "构建器" begin
    using SparseArrays

    mat = sprand(10, 20, 0.3)
    rel = DirectRelation(mat, collect(1:10), collect(1:20), :rating)

    # ==========================================
    # ItemSimilarityBuilder
    # ==========================================
    @testset "ItemSimilarityBuilder" begin
        builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(5))
        graph = build(builder, rel)

        @test graph isa CachedSimilarityGraph
        @test size(graph.matrix) == (20, 20)

        neighbors = get_neighbors(graph, 1)
        @test length(neighbors) <= 5
        @test neighbors[1][1] == 1  # 自身最近

        # 缓存命中
        neighbors2 = get_neighbors(graph, 1)
        @test neighbors == neighbors2
    end

    # ==========================================
    # UserSimilarityBuilder
    # ==========================================
    @testset "UserSimilarityBuilder" begin
        builder = UserSimilarityBuilder(PearsonSimilarity(), TopKNeighbors(3))
        graph = build(builder, rel)

        @test graph isa CachedSimilarityGraph
        @test size(graph.matrix) == (10, 10)

        neighbors = get_neighbors(graph, 1)
        @test length(neighbors) <= 3
    end

    # ==========================================
    # 默认参数
    # ==========================================
    @testset "默认参数" begin
        b1 = ItemSimilarityBuilder()
        @test b1.metric isa CosineSimilarity
        @test b1.pruning isa TopKNeighbors
        @test b1.pruning.k == 50

        b2 = UserSimilarityBuilder()
        @test b2.metric isa CosineSimilarity
        @test b2.pruning.k == 50
    end

    # ==========================================
    # SimilarityGraph 非缓存版本
    # ==========================================
    @testset "SimilarityGraph 基础版" begin
        builder = ItemSimilarityBuilder(CosineSimilarity(), MinSimilarityThreshold(0.1))
        wrapper = ItemUserMatrix(rel.matrix)
        sim = compute_similarity(builder.metric, wrapper, builder.pruning)
        graph = SimilarityGraph(sim, builder.metric)

        @test graph isa SimilarityGraph
        @test graph isa InferredRelation
        neighbors = get_neighbors(graph, 1)
        @test !isempty(neighbors)
    end
end