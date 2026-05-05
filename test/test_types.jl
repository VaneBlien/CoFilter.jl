@testset "类型层级" begin
    using SparseArrays
    # ==========================================
    # 相似度度量层级
    # ==========================================
    @test CosineSimilarity <: SimilarityMetric
    @test PearsonSimilarity <: SimilarityMetric
    @test JaccardSimilarity <: SimilarityMetric
    @test AdjustedCosineSimilarity <: SimilarityMetric

    # 带参数构造
    metric = AdjustedCosineSimilarity(0.3)
    @test metric.damping_factor == 0.3

    # 默认参数
    metric_default = AdjustedCosineSimilarity()
    @test metric_default.damping_factor == 0.5

    # 参数校验
    @test_throws ArgumentError AdjustedCosineSimilarity(-0.1)
    @test_throws ArgumentError AdjustedCosineSimilarity(1.5)

    # ==========================================
    # 裁剪策略层级
    # ==========================================
    @test TopKNeighbors <: PruningStrategy
    @test MinSimilarityThreshold <: PruningStrategy

    pruning_k = TopKNeighbors(50)
    @test pruning_k.k == 50
    @test_throws ArgumentError TopKNeighbors(0)
    @test_throws ArgumentError TopKNeighbors(-1)

    pruning_t = MinSimilarityThreshold(0.2)
    @test pruning_t.threshold == 0.2
    @test_throws ArgumentError MinSimilarityThreshold(-0.1)
    @test_throws ArgumentError MinSimilarityThreshold(1.5)

    # ==========================================
    # 关系层级
    # ==========================================
    @test DirectRelation <: AbstractRelation
    @test InferredRelation <: AbstractRelation
    @test SimilarityGraph <: InferredRelation
    @test CachedSimilarityGraph <: InferredRelation

    # ==========================================
    # 构建器层级
    # ==========================================
    @test UserSimilarityBuilder <: RelationBuilder
    @test ItemSimilarityBuilder <: RelationBuilder

    # 默认构造
    b1 = UserSimilarityBuilder()
    @test b1.metric isa CosineSimilarity
    @test b1.pruning isa TopKNeighbors
    @test b1.pruning.k == 50

    # 自定义构造
    b2 = ItemSimilarityBuilder(PearsonSimilarity(), MinSimilarityThreshold(0.3))
    @test b2.metric isa PearsonSimilarity
    @test b2.pruning isa MinSimilarityThreshold
    @test b2.pruning.threshold == 0.3

    # ==========================================
    # 推荐引擎层级
    # ==========================================
    @test UserBasedRecommender <: AbstractRecommender
    @test ItemBasedRecommender <: AbstractRecommender
    @test HybridRecommender <: AbstractRecommender

    # ==========================================
    # 融合策略层级
    # ==========================================
    @test WeightedSum <: FusionStrategy
    @test RoundRobin <: FusionStrategy

    w = WeightedSum([0.6, 0.4])
    @test w.weights == [0.6, 0.4]

    r = RoundRobin()
    @test r isa RoundRobin

    # ==========================================
    # 矩阵视角包装
    # ==========================================
    m = sprand(5, 10, 0.3)
    uw = UserItemMatrix(m)
    iw = ItemUserMatrix(m)
    @test uw.data === m
    @test iw.data === m
end