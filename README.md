
# CoFilter.jl

[中文文档](readme_cn.md)

**Pure Julia collaborative filtering, built around sparse matrices and multiple dispatch.**

## What CoFilter does

Given a user-item interaction matrix, CoFilter builds similarity graphs and produces Top-N recommendations. It supports user-based CF, item-based CF, and hybrid fusion — all sharing the same underlying sparse matrix infrastructure, with no Python dependencies.

## Design choices

- **Directly on `SparseMatrixCSC`**. Similarity computation calls BLAS3 on the native sparse format. No conversion to dense, no copies. See `src/similarity/computation.jl`.
- **Algorithms as concrete types**. `UserBasedRecommender`, `ItemBasedRecommender`, and `HybridRecommender` are distinct types. Dispatch selects the right code path at compile time — no runtime `if` chains.
- **Builders separate metric from strategy**. A `Builder` pairs a similarity metric (Cosine, Pearson, Jaccard, AdjustedCosine) with a pruning strategy (TopK, threshold). This means any metric can be combined with any pruning without changing recommendation logic. See `src/similarity/builders.jl`.
- **Incremental update touches only affected rows**. New interactions don't trigger full recomputation. See `src/system/update.jl`.
- **Cached neighbor lookups**. `SimilarityGraph` has an optional `CachedSimilarityGraph` variant that stores neighbor lists after first access. See `src/graph/cache.jl`.

## Quick start

```julia
using CoFilter, SparseArrays

# 100 users, 200 items, 5% density
matrix = sprand(100, 200, 0.05)
direct_rel = DirectRelation(matrix, collect(1:100), collect(1:200), :rating)

# Item-based CF with cosine similarity, top-30 neighbors
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
sys = RecommendationSystem(direct_rel, Dict(:item_sim => builder), ItemBasedRecommender())

train!(sys)
top10 = recommend(sys, 1, 10)
```

## Supported algorithms

| Type | Constructor |
|------|-------------|
| User-Based CF | `UserBasedRecommender()` |
| Item-Based CF | `ItemBasedRecommender()` |
| Weighted Hybrid | `HybridRecommender(engines, WeightedSum(weights))` |
| Round-Robin Hybrid | `HybridRecommender(engines, RoundRobin())` |

## Similarity metrics

`CosineSimilarity()`, `PearsonSimilarity()`, `JaccardSimilarity()`, `AdjustedCosineSimilarity(damping_factor)`

## Pruning strategies

`TopKNeighbors(k)`, `MinSimilarityThreshold(t)`

## Evaluation metrics

Precision@K, Recall@K, NDCG@K, Hit Rate@K — see `src/evaluation/metrics.jl`

## Testing

```julia
using Pkg
Pkg.test("CoFilter")
```

225 tests across all modules.

## Source layout

```
src/
├── CoFilter.jl
├── core/           # AbstractRelation, DirectRelation, interfaces
├── similarity/     # Metrics, pruning, computation, builders, distributed
├── recommenders/   # User-based, item-based, hybrid, fusion strategies
├── graph/          # SimilarityGraph, CachedSimilarityGraph
├── system/         # RecommendationSystem, training, incremental update
├── evaluation/     # Train/test split, metrics, cross-validation
└── utils/          # Sparse utilities, cold-start fallbacks
```

## License

MIT
