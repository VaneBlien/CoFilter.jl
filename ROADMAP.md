# CoFilter.jl - Roadmap & TODO

---

## Algorithm Extensions

| Priority | Task | Description |
|----------|------|-------------|
| High | **Matrix Factorization Builder** | Implement SVD/ALS for sparse matrices, especially for implicit feedback scenarios. |
| High | **Content-Based Builder** | Support item features (type, description, etc.) to solve item cold-start. |
| Medium | **Time Decay Builder** | Wrap existing builders; assign higher weights to recent interactions. |
| Medium | **Penalty Builder** | Down-weight popular items to improve recommendation diversity. |
| Low | **Binary/Implicit Feedback Support** | Optimize handling of clicks, purchases, and other implicit signals beyond explicit ratings. |

## Performance Optimization

| Priority | Task | Description |
|----------|------|-------------|
| High | **Multi-Process Distribution** | Leverage `Distributed.jl` for true multi-worker parallel similarity computation. |
| Medium | **GPU Acceleration** | Use CUDA.jl to accelerate matrix multiplications. |
| Medium | **Incremental Update Optimization** | Currently new users/items trigger full recomputation; implement true local updates. |
| Low | **Memory Optimization** | Out-of-core processing for very large matrices. |

## Engineering Improvements

| Priority | Task | Description |
|----------|------|-------------|
| High | **Serialization/Deserialization** | Save/load trained models to avoid recomputation on every startup. |
| High | **Logging System** | Replace `println` with proper leveled logging and file output. |
| Medium | **Configuration System** | Drive system setup via TOML/YAML config files. |
| Medium | **REST API** | Wrap recommendation service endpoints using HTTP.jl. |
| Low | **Documentation Site** | Auto-generated API docs via Documenter.jl. |
| Low | **Package Registration** | Submit to Julia General Registry for `Pkg.add` support. |

## Testing & Quality

| Priority | Task | Description |
|----------|------|-------------|
| High | **MovieLens End-to-End Integration Test** | Automated workflow: download → train → evaluate → recommend. |
| Medium | **Performance Regression Tests** | Ensure new commits don't degrade speed. |
| Medium | **Edge Case Coverage** | Very large `n`, empty matrices, all-zero rows, etc. |
| Low | **Code Coverage** | Target 90%+ coverage. |

## Documentation

| Priority | Task | Description |
|----------|------|-------------|
| High | **Complete Examples** | `custom_metric.jl`, `movielens_demo.jl`, and other full usage examples. |
| Medium | **API Reference** | Usage guide for every public function. |
| Low | **Performance Tuning Guide** | Recommendations for choosing `TopK`, `batch_size`, and other parameters. |

---

**Current Version**: v0.1.0  
**Test Suite**: 225 tests, all passing  
**Supported Algorithms**: User-Based CF, Item-Based CF, Hybrid (WeightedSum / RoundRobin)  
**Similarity Metrics**: Cosine, Pearson, Jaccard, Adjusted Cosine
