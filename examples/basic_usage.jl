using CoFilter
using SparseArrays

# ============================================================
# 基础使用示例
# ============================================================

# 1. 准备模拟数据
n_users, n_items = 100, 200
matrix = sprand(n_users, n_items, 0.05)
user_ids = 1:n_users
item_ids = 1:n_items

# 2. 构建直接关系
direct_rel = DirectRelation(matrix, user_ids, item_ids, :rating)

# 3. 配置系统：基于物品的协同过滤
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
sys = RecommendationSystem(
    direct_rel,
    Dict(:item_sim => builder),
    ItemBasedRecommender()
)

# 4. 训练
train!(sys)

# 5. 为用户 1 推荐 10 个物品
top10 = recommend(sys, 1, 10)
println("为用户 1 推荐: $top10")
