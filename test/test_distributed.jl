@testset "分布式计算" begin
    using SparseArrays
    using Distributed

    # ==========================================
    # 分块函数
    # ==========================================
    @testset "分块函数" begin
        chunks = CoFilter._split_indices(100, 4)
        @test length(chunks) == 4
        @test chunks[1] == 1:25
        @test chunks[end] == 76:100

        chunks2 = CoFilter._split_indices(7, 3)
        @test length(chunks2) == 3
        @test chunks2[1] == 1:3
        @test chunks2[end] == 7:7
    end

    # ==========================================
    # 分布式计算（单 worker 降级测试）
    # ==========================================
    @testset "单 worker 降级" begin
        mat = sprand(20, 30, 0.2)
        metric = CosineSimilarity()
        pruning = TopKNeighbors(5)

        sim = CoFilter.compute_similarity_distributed(
            metric, mat, pruning; n_workers = 1
        )

        @test size(sim) == (20, 20)
        for i in 1:20
            @test nnz(sim[i, :]) <= 5
        end
    end

    # ==========================================
    # 自动选择模式
    # ==========================================
    @testset "自动选择模式" begin
        mat_small = sprand(100, 200, 0.1)
        metric = CosineSimilarity()
        pruning = TopKNeighbors(10)

        sim_small = CoFilter.compute_similarity_auto(
            metric, mat_small, pruning; threshold = 1000, n_workers = 1
        )
        @test size(sim_small) == (100, 100)
    end

    # ==========================================
    # 分布式结果与单机结果一致
    # ==========================================
    @testset "结果一致性" begin
        mat = sprand(50, 40, 0.15)
        metric = CosineSimilarity()
        pruning = TopKNeighbors(10)

        sim_single = compute_similarity(metric, UserItemMatrix(mat), pruning)

        sim_dist = CoFilter.compute_similarity_distributed(
            metric, mat, pruning; n_workers = 1
        )

        @test size(sim_single) == size(sim_dist)
        @test nnz(sim_single) == nnz(sim_dist)
    end
end