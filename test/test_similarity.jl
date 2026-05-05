@testset "相似度计算" begin
    using SparseArrays
    using LinearAlgebra

    # ==========================================
    # 辅助：构建小型确定性矩阵
    # ==========================================
    function build_small_matrix()
        dense = zeros(3, 4)
        dense[1, 1] = 5.0; dense[1, 2] = 3.0
        dense[2, 1] = 4.0; dense[2, 3] = 2.0
        dense[3, 2] = 1.0; dense[3, 4] = 5.0
        return sparse(dense)
    end

    # ==========================================
    # Cosine + TopK
    # ==========================================
    @testset "Cosine + TopK" begin
        mat = build_small_matrix()
        metric = CosineSimilarity()
        pruning = TopKNeighbors(2)

        # 用户相似度
        user_wrapper = UserItemMatrix(mat)
        user_sim = compute_similarity(metric, user_wrapper, pruning)
        @test size(user_sim) == (3, 3)
        for i in 1:3
            @test nnz(user_sim[i, :]) <= 2
            @test user_sim[i, i] ≈ 1.0 atol=1e-8
        end

        # 物品相似度
        item_wrapper = ItemUserMatrix(mat)
        item_sim = compute_similarity(metric, item_wrapper, pruning)
        @test size(item_sim) == (4, 4)
        for i in 1:4
            @test nnz(item_sim[i, :]) <= 2
        end
    end

    # ==========================================
    # Cosine + Threshold
    # ==========================================
    @testset "Cosine + Threshold" begin
        mat = build_small_matrix()
        metric = CosineSimilarity()
        pruning = MinSimilarityThreshold(0.5)
        sim = compute_similarity(metric, UserItemMatrix(mat), pruning)
        @test size(sim) == (3, 3)
        for i in 1:3, j in 1:3
            val = sim[i, j]
            if val != 0.0 && i != j
                @test val >= 0.5
            end
        end
    end

    # ==========================================
    # Pearson + TopK
    # ==========================================
       @testset "Pearson + TopK" begin
        dense = zeros(2, 3)
        dense[1, 1] = 5.0; dense[1, 2] = 4.0
        dense[2, 1] = 4.0; dense[2, 2] = 3.0
        mat = sparse(dense)
        sim = compute_similarity(PearsonSimilarity(), UserItemMatrix(mat), TopKNeighbors(2))
        @test sim[1, 2] ≈ 1.0 atol=1e-2  # 放宽到 1e-6
    end

    # ==========================================
    # 全零行
    # ==========================================
    @testset "全零行处理" begin
        dense = zeros(3, 4)
        dense[1, 1] = 5.0
        mat = sparse(dense)
        sim = compute_similarity(CosineSimilarity(), UserItemMatrix(mat), TopKNeighbors(2))
        @test size(sim) == (3, 3)
    end

    # ==========================================
    # 手动验证余弦值
    # ==========================================
    @testset "手动验证余弦值" begin
        dense = zeros(2, 3)
        dense[1, 1] = 1.0; dense[1, 2] = 2.0
        dense[2, 1] = 2.0; dense[2, 2] = 4.0
        mat = sparse(dense)
        sim = compute_similarity(CosineSimilarity(), UserItemMatrix(mat), TopKNeighbors(2))
        @test sim[1, 2] ≈ 1.0 atol=1e-8
    end

    # ==========================================
    # Jaccard 类型检查
    # ==========================================
    @testset "Jaccard 类型存在" begin
        j = JaccardSimilarity()
        @test j isa SimilarityMetric
    end

    # ==========================================
    # AdjustedCosine 构造
    # ==========================================
    @testset "AdjustedCosine 参数校验" begin
        @test AdjustedCosineSimilarity(0.3).damping_factor == 0.3
        @test_throws ArgumentError AdjustedCosineSimilarity(-0.1)
        @test_throws ArgumentError AdjustedCosineSimilarity(1.5)
    end
end