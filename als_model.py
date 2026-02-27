train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"训练集样本数: {train.count()}, 测试集样本数: {test.count()}")
#ALS 模型训练
als = ALS(maxIter=10, rank=10, regParam=0.1,
    userCol="user_id", itemCol="item_id", ratingCol="rating",
    coldStartStrategy="drop")
model_als=als.fit(train)
print("ALS 模型训练完成！")
#ALS 推荐：为每个用户生成 Top-10 推荐
user_recs_als = model_als.recommendForAllUsers(10)
# 展开推荐结果（把数组变成多行）
user_recs_exp_als = user_recs_als.select("user_id", explode("recommendations").alias("rec")) \
      .select("user_id", col("rec.item_id").alias("item_id"), col("rec.rating").alias("pred_score"))
print("ALS 推荐结果示例：")
user_recs_exp_als.show(5)
#热门推荐基线
#计算每个商品的总评分（热度）
item_popularity = train.groupBy("item_id").agg(F.sum("rating").alias("total_score"))
# 10.2 选出最热的 10 个商品
top_items = item_popularity.orderBy(F.desc("total_score")).limit(10).select("item_id").collect()
hot_items = [row.item_id for row in top_items]
print(f"热门商品 Top10: {hot_items}")
#为测试集中的每个用户生成同样的热门推荐（用于评估）
test_users = test.select("user_id").distinct()
# 创建热门商品的 DataFrame
hot_items_df = spark.createDataFrame([(item_id,) for item_id in hot_items], ["item_id"])
# 笛卡尔积：每个用户 × 每个热门商品
hot_recs_test = test_users.crossJoin(hot_items_df)
# 给每个推荐加一个占位评分（用于统一格式）
hot_recs_test = hot_recs_test.withColumn("pred_score", F.lit(1.0))
print("热门推荐生成完成，示例：")
hot_recs_test.show(5)
#评估函数（计算准确率和召回率）
def evaluate_metrics(recommendations, test):
    """
    recommendations: DataFrame with columns user_id, item_id, pred_score
    test: DataFrame with columns user_id, item_id, rating
    """
    #从测试集中找出用户真正喜欢的商品（评分 >= 3）
    test_likes = test.filter(F.col("rating") >= 3) \
      .groupBy("user_id") \
      .agg(F.collect_set("item_id").alias("true_items"))
    #从推荐结果中收集每个用户的推荐商品集合
    rec_items = recommendations.groupBy("user_id") \
      .agg(F.collect_set("item_id").alias("rec_items"))
    #合并两个集合
    eval_df = test_likes.join(rec_items, on="user_id", how="inner")
    #定义计算单个用户指标的 UDF
    def metrics_func(true, rec):
        true_set = set(true)
        rec_set = set(rec)
        hits = len(true_set.intersection(rec_set))
        prec = hits / len(rec_set) if rec_set else 0.0
        recall = hits / len(true_set) if true_set else 0.0
        return (prec, recall)
    #注册 UDF
    metrics_udf = F.udf(metrics_func, "struct<precision:float,recall:float>")
    eval_df = eval_df.withColumn("metrics", metrics_udf(F.col("true_items"), F.col("rec_items")))

    #计算平均准确率和召回率
    avg_prec = eval_df.agg(F.avg("metrics.precision")).collect()[0][0]
    avg_recall = eval_df.agg(F.avg("metrics.recall")).collect()[0][0]
    return avg_prec, avg_recall
#评估 ALS 模型
prec_als, recall_als = evaluate_metrics(user_recs_exp_als, test)
print(f"ALS 平均准确率: {prec_als:.4f}, 平均召回率: {recall_als:.4f}")
#评估热门推荐
prec_hot, recall_hot = evaluate_metrics(hot_recs_test, test)
print(f"热门推荐 平均准确率: {prec_hot:.4f}, 平均召回率: {recall_hot:.4f}")
#计算覆盖率
total_items = df.select("item_id").distinct().count()
rec_items_count = user_recs_exp_als.select("item_id").distinct().count()
coverage = rec_items_count / total_items
print(f"ALS 推荐覆盖率: {coverage:.2%}")
#生成商品热度 CSV（供本地可视化使用）
item_pop = df.groupBy("item_id").agg(
    F.mean("rating").alias("avg_rating"),
    F.count("rating").alias("pop_count")
)
# 合并成一个文件并保存
item_pop.coalesce(1).write.csv("/content/item_pop.csv", header=True, mode="overwrite")
print("商品热度 CSV 已生成，请下载：/content/item_pop.csv")
#下载 CSV 到本地（如果需要）
import os
from google.colab import files
#找到生成的文件
files_in_dir = os.listdir("/content/item_pop.csv")
csv_file = [f for f in files_in_dir if f.endswith('.csv')][0]
files.download(f"/content/item_pop.csv/{csv_file}")
print("全部完成！")
