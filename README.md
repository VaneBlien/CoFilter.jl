
# CoFilter.jl
[中文文档](readme_cn.md)


**A flexible, high-performance collaborative filtering module in pure Julia, built on multiple dispatch and a relational modular architecture.**

## Features

- **Relation Abstraction**: User-item interactions and similarity graphs are unified as "relation" types.
- **Algorithms as Types**: Every similarity metric and recommendation strategy is a concrete Julia type.
- **Multiple Dispatch Driven**: Zero runtime overhead for strategy selection.
- **High Performance**: Operates directly on sparse matrices with BLAS3-accelerated multiplications.
- **Incremental Updates**: Hot-update support for new interaction data without full retraining.
- **Offline Evaluation**: Built-in Precision, Recall, NDCG, and Hit Rate metrics.

## Architecture

```
Data Layer (Relations) → Computation Layer (Builders) → Strategy Layer (Recommenders) → Top-N Results
```

The module is organized into a clean, layered structure that separates concerns and maximizes extensibility:

```
src/
├── CoFilter.jl                  # Module entry point
├── core/                        # Core types & abstractions
│   ├── interfaces.jl
│   ├── types.jl
│   └── relations.jl
├── similarity/                  # Similarity computation
│   ├── metrics.jl               #   Cosine, Pearson, Jaccard, AdjustedCosine
│   ├── pruning.jl               #   TopK, MinSimilarityThreshold
│   ├── computation.jl           #   Core computation kernels
│   ├── builders.jl              #   Builder pattern for assembling pipelines
│   └── distributed.jl           #   Multi-worker distributed computation
├── recommenders/                # Recommendation engines
│   ├── base.jl
│   ├── user_based.jl
│   ├── item_based.jl
│   ├── fusion.jl                #   WeightedSum & RoundRobin strategies
│   └── hybrid.jl
├── graph/                       # Similarity graph representation
│   ├── similarity_graph.jl
│   └── cache.jl                 #   Transparent neighbor caching
├── system/                      # System assembly & lifecycle
│   ├── recommendation_system.jl
│   ├── training.jl              #   Full training
│   └── update.jl                #   Incremental online updates
├── evaluation/                  # Offline evaluation suite
│   ├── splitting.jl             #   Train/test split strategies
│   ├── metrics.jl               #   Precision, Recall, NDCG, Hit Rate
│   └── cross_validation.jl
└── utils/                       # Utilities
    ├── sparse_utils.jl
    ├── cold_start.jl            #   Fallback strategies for new users/items
    └── validation.jl
```

This modular design allows you to swap similarity metrics, pruning strategies, and recommendation engines independently, or combine them into hybrid systems.

## Quick Start

### Installation

The package is not yet registered. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/VaneBlien/CoFilter.jl")
```

Or clone and load via local path:

```julia
using Pkg
Pkg.develop(path="path/to/CoFilter.jl")
```

### Basic Usage

```julia
using CoFilter, SparseArrays

# 1. Prepare data: 100 users, 200 items, 5% density
n_users, n_items = 100, 200
matrix = sprand(n_users, n_items, 0.05)
direct_rel = DirectRelation(matrix, collect(1:n_users), collect(1:n_items), :rating)

# 2. Configure an item-based CF system
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
sys = RecommendationSystem(direct_rel, Dict(:item_sim => builder), ItemBasedRecommender())

# 3. Train
train!(sys)

# 4. Get top-10 recommendations for user 1
top10 = recommend(sys, 1, 10)
println("Recommendations for user 1: $top10")
```

### Offline Evaluation

```julia
# Split data
train_rel, test_pairs = train_test_split(direct_rel, 0.2)

# Evaluate with a factory closure
metrics = evaluate(() -> begin
    sys = RecommendationSystem(train_rel, Dict(:item_sim => builder), engine)
    train!(sys)
    return sys
end, test_pairs, 10)

println("Precision@10: $(metrics.precision)")
println("Recall@10:    $(metrics.recall)")
println("NDCG@10:      $(metrics.ndcg)")
```

### MovieLens Demo

A full end-to-end example on the built-in MovieLens 100K dataset:

```julia
include("examples/movielens_demo.jl")
```

## Supported Algorithms

| Category | Algorithm | Type |
|----------|-----------|------|
| Collaborative Filtering | User-Based CF | `UserBasedRecommender` |
| Collaborative Filtering | Item-Based CF | `ItemBasedRecommender` |
| Hybrid | Weighted Fusion | `HybridRecommender` + `WeightedSum` |
| Hybrid | Round-Robin Fusion | `HybridRecommender` + `RoundRobin` |

## Similarity Metrics

- `CosineSimilarity` — Cosine similarity
- `PearsonSimilarity` — Pearson correlation coefficient
- `JaccardSimilarity` — Jaccard index
- `AdjustedCosineSimilarity` — Adjusted cosine with configurable damping

## Pruning Strategies

- `TopKNeighbors(k)` — Keep only the `k` nearest neighbors
- `MinSimilarityThreshold(t)` — Keep neighbors with similarity ≥ `t`

## Evaluation Metrics

- Precision@K
- Recall@K
- NDCG@K
- Hit Rate@K

## Testing

Run the full test suite (225 tests across all modules):

```julia
using Pkg
Pkg.test("CoFilter")
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned algorithm extensions, performance optimizations, and engineering improvements.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
