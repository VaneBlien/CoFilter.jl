using CoFilter
using SparseArrays
using Downloads
using DelimitedFiles

# ============================================================
# 1. 下载并解压 MovieLens 100K 数据集
# ============================================================
const DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
const ZIP_PATH = "ml-100k.zip"
const DATA_DIR  = "ml-100k"
const DATA_FILE = joinpath(DATA_DIR, "u.data")

if !isfile(DATA_FILE)
    println("正在下载 MovieLens 100K 数据集...")
    Downloads.download(DATA_URL, ZIP_PATH)
    println("下载完成，正在解压...")
    run(`powershell -Command "Expand-Archive -Path '$ZIP_PATH' -DestinationPath '.'"`)
    println("解压完成！")
else
    println("数据文件已存在，跳过下载。")
end

# ============================================================
# 2. 加载并解析数据
# ============================================================
println("正在加载数据...")

raw_data = readdlm(DATA_FILE, '\t', Float64)

user_ids_raw = Int.(raw_data[:, 1])
item_ids_raw = Int.(raw_data[:, 2])
ratings_raw  = raw_data[:, 3]

unique_users = sort(unique(user_ids_raw))
unique_items = sort(unique(item_ids_raw))

user_to_idx = Dict{Int, Int}(zip(unique_users, 1:length(unique_users)))
item_to_idx = Dict{Int, Int}(zip(unique_items, 1:length(unique_items)))

num_users = length(unique_users)
num_items = length(unique_items)

println("用户数: $num_users, 电影数: $num_items, 评分数: $(length(user_ids_raw))")

# 构建稀疏评分矩阵
I = [user_to_idx[uid] for uid in user_ids_raw]
J = [item_to_idx[iid] for iid in item_ids_raw]
rating_matrix = sparse(I, J, ratings_raw, num_users, num_items)

# ============================================================
# 3. 创建 CoFilter DirectRelation
# ============================================================
println("正在构建 CoFilter 数据关系...")
direct_relation = DirectRelation(
    rating_matrix,
    collect(1:num_users),
    collect(1:num_items),
    :rating
)

# ============================================================
# 4. 划分训练集与测试集
# ============================================================
println("正在划分数据集...")
train_relation, test_pairs = train_test_split(direct_relation, 0.2; random_seed=42)
println("训练集评分数: $(nnz(train_relation.matrix))")
println("测试集样本数: $(length(test_pairs))")

# ============================================================
# 5. 配置并训练
# ============================================================
println("\n正在训练模型...")
builder = ItemSimilarityBuilder(CosineSimilarity(), TopKNeighbors(30))
engine = ItemBasedRecommender()

sys = RecommendationSystem(train_relation, Dict(:item_sim => builder), engine)
train!(sys)

# ============================================================
# 6. 离线评估
# ============================================================
println("\n正在评估模型性能...")
test_metrics = evaluate(() -> begin
    eval_sys = RecommendationSystem(train_relation, Dict(:item_sim => builder), engine)
    train!(eval_sys)
    return eval_sys
end, test_pairs, 10)

println("\n========== 最终评估结果 (Top-10) ==========")
println("精确率 (Precision): $(round(test_metrics.precision, digits=4))")
println("召回率 (Recall):    $(round(test_metrics.recall, digits=4))")
println("命中率 (Hit Rate):  $(round(test_metrics.hit_rate, digits=4))")
println("NDCG:               $(round(test_metrics.ndcg, digits=4))")

# ============================================================
# 7. 实际推荐示例
# ============================================================
println("\n正在生成推荐...")
example_user_id = 1
recommended_items = recommend(sys, example_user_id, 10)
println("为用户 $example_user_id 推荐的电影ID: $recommended_items")