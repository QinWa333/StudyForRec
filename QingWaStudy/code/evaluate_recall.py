import pandas as pd
import json
import os
import logging
from collections import defaultdict

# 配置日志
# 配置log日志
logging.basicConfig(
    filename='../log/evaluate_recall.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 输出到控制台的同时也记录到日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_recall_result(recall_path):
    """
    加载召回结果
    """
    try:
        with open(recall_path, 'r') as f:
            recall_result = json.load(f)

        # 将用户ID转换为int
        recall_result = {int(k): v for k, v in recall_result.items()}

        logger.info(f"成功加载召回结果，包含 {len(recall_result)} 个用户")
        return recall_result

    except Exception as e:
        logger.error(f"加载召回结果失败: {e}")
        return {}


def calculate_recall_at_k(recall_result, test_dict, k_values=[10, 30, 50, 100]):
    """
    计算不同k值下的召回率

    参数:
    recall_result: 召回结果字典 {user_id: [(item_id, score), ...]}
    test_dict: 测试数据字典 {user_id: 最后一次观看的电影ID}
    k_values: 需要计算的k值列表

    返回:
    每个k值对应的召回率字典
    """
    recall_scores = defaultdict(float)
    valid_users = 0

    for user_id, last_movie in test_dict.items():
        # 检查用户是否在召回结果中
        if user_id in recall_result:
            # 获取该用户的召回列表
            rec_items = [item_id for item_id, _ in recall_result[user_id]]

            # 对每个k值计算召回率
            for k in k_values:
                # 取前k个召回物品
                top_k_items = rec_items[:k]
                # 如果测试电影在召回列表中，命中数为1，否则为0
                hit = 1 if last_movie in top_k_items else 0
                # 累加命中数
                recall_scores[k] += hit

            valid_users += 1

    # 计算最终召回率（命中数/总用户数）
    for k in k_values:
        recall_scores[k] = recall_scores[k] / valid_users if valid_users > 0 else 0
        logger.info(f"召回率@ {k}: {recall_scores[k]:.4f}")

    return dict(recall_scores)


def main():
    # 配置路径
    base_path = '../dataset/recall_result/ItemCF'
    recall_path = os.path.join(base_path, 'recall_result.json')
    logger.info('验证集用户召回率：（考虑物品在点击序列中的距离）（打压活跃用户）')
    test_click_path = '../dataset/processed_df/validation.parquet'  # 请根据实际路径修改
    # 加载数据
    test_last_click_df_ = pd.read_parquet(test_click_path)
    test_user_movie_df = test_last_click_df_[['user_id', 'movie_id']]
    test_user_movie_dict = test_user_movie_df.set_index('user_id')['movie_id'].to_dict()

    recall_result = load_recall_result(recall_path)

    # 计算召回率
    recall_at_k = calculate_recall_at_k(recall_result, test_user_movie_dict, k_values=[10, 30, 50, 100])

    # 输出结果
    print("\n===== 召回率结果 =====")
    for k, score in recall_at_k.items():
        print(f"Recall@{k}: {score:.4f}")

    # 保存结果
    result_path = '../dataset/evaluate_recall/evaluate_itemCF.json'
    with open(result_path, 'w') as f:
        json.dump(recall_at_k, f, indent=2)
    logger.info(f"召回率指标已保存到 {result_path}")


if __name__ == "__main__":
    main()