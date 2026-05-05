using BenchmarkTools
using CoFilter
using SparseArrays

println("="^50)
println("  相似度计算性能基准测试")
println("="^50)

# 小型矩阵
println("\n--- 小型矩阵 (500 × 800) ---")
mat_small = sprand(500, 800, 0.05)
metric = CosineSimilarity()
pruning = TopKNeighbors(20)

@btime compute_similarity($metric, UserItemMatrix($mat_small), $pruning)

# 中型矩阵
println("\n--- 中型矩阵 (2000 × 3000) ---")
mat_medium = sprand(2000, 3000, 0.03)

@btime compute_similarity($metric, UserItemMatrix($mat_medium), $pruning)

# 大型矩阵
println("\n--- 大型矩阵 (5000 × 5000) ---")
mat_large = sprand(5000, 5000, 0.01)

@btime compute_similarity($metric, UserItemMatrix($mat_large), $pruning)