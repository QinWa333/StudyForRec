import os
import numpy as np
import json
from tqdm import  tqdm
import logging
import pandas as pd
# 配置log日志
logging.basicConfig(
    filename='../../log/002ItemCF_recall.log',
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

def recall(users,user_item_dict,sim_dict,recall_num=100,nearest_num=10,top_n=200):
    """

    :param users: 用户id列表
    :param user_item_dict:
    :param sim_dict:
    :param recall_num: 每个用户的召回数量（不足时用热门电影补全）
    :param nearest_num: 取用户最近看过的电影数作为观看序列
    :param top_n: 为每个观看序列中的电影取多少个相似电影
    :return:
    """
    user_rec_dict = {}
    # 为每个用户生成推荐
    for user in tqdm(users):

        rec_items_score = {}
        # 获取用户点击序列，并反转使得靠前的为最近观看的
        user_click_items = user_item_dict[user]
        user_click_items = user_click_items[::-1][:nearest_num]

        for loc, item in enumerate(user_click_items):
            # 取两百个最相似物品
            for related_item,sim_score in sorted(sim_dict[item].items(), key=lambda x:x[1],reverse=True)[0:top_n]:
                if related_item not in user_click_items:
                    rec_items_score.setdefault(related_item,0)
                    rec_items_score[related_item] += sim_score * (0.7**loc)
        rec_items = sorted(rec_items_score.items(),key=lambda x:x[1],reverse=True)[:recall_num]
        # 保存该用户的召回结果
        user_rec_dict[user] = rec_items


    return user_rec_dict



if __name__ =="__main__":

    #读取相关文件
    ## 用户表生成用户列表
    user_df = pd.read_parquet('../../dataset/processed_df/users.parquet')
    user_id_list = user_df['user_id'].to_list()

    ## 读取用户点击序列和相似度字典
    save_path = '../../tmp_result/ItemCF'
    file_path = os.path.join(save_path,'user_item_dict.json')

    with open(file_path,'r')as f:
        loaded_user_item_dict = json.load(f)

    user_item_dict = {int(k): v for k, v in loaded_user_item_dict.items()}
    logger.info('成功读取用户点击序列')


    file_path = os.path.join(save_path,'item_similarity.json')
    with open(file_path,'r')as f:
       loaded_item_sim_dict = json.load(f)

    item_sim_dict = {int(k):dict(v) for k,v in loaded_item_sim_dict.items()}
    logger.info('成功读取物品相似度字典')

    '''   
    sample_item = next(iter(item_sim_dict))  # 取第一个物品作为示例
    logger.info(f"物品 {sample_item} 的前5个相似物品:")
    for related_item, score in sorted(item_sim_dict[sample_item].items(),
                                      key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  - {related_item}: {score:.4f}")
    '''

    valid_users = [user for user in user_id_list if user in user_item_dict]
    print(f"总用户数: {len(user_id_list)}, 有交互记录的用户数: {len(valid_users)}")

    recall_result = recall(user_id_list,user_item_dict,item_sim_dict,recall_num=100,top_n=200,nearest_num=10)
    logger.info('成功生成用户推荐列表')

    ######################## test ###########################
    # 验证基本结构
    logger.info(f"召回结果类型: {type(recall_result)}\n")  # 应输出 <class 'dict'>
    logger.info(f"召回的用户数量: {len(recall_result)}\n")  # 应等于 user_id_list 的长度

    # 统计每个用户的召回数量
    recall_counts = [len(items) for items in recall_result.values()]
    min_recall = min(recall_counts)
    max_recall = max(recall_counts)
    avg_recall = sum(recall_counts) / len(recall_counts)

    logger.info(f"最少召回物品数: {min_recall}")
    logger.info(f"最多召回物品数: {max_recall}")
    logger.info(f"平均召回物品数: {avg_recall:.2f}")

    # 找出召回最少的用户示例
    min_users = [user_id for user_id, items in recall_result.items() if len(items) == min_recall]
    logger.info(f"召回最少的用户示例: {min_users[:5]} (共{len(min_users)}个用户)")

    # 检查第一个用户的召回结果
    sample_user = next(iter(recall_result))
    sample_recs = recall_result[sample_user]
    logger.info(f"\n用户 {sample_user} 的召回结果:\n")
    logger.info(f"  推荐数量: {len(sample_recs)}\n")  # 应小于等于 recall_num
    logger.info(f"  格式示例: {sample_recs[:3]}\n")  # 应输出 [(物品ID, 分数), ...]

    # 验证分数是否按降序排列
    if len(sample_recs) > 1:
        scores = [score for _, score in sample_recs]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "分数未按降序排列"
        logger.info("  分数验证: ✅ 降序排列\n")

    save_path = '../../dataset/recall_result/ItemCF'
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, 'recall_result.json')
    try:
        # 转换为可JSON序列化的格式
        json_recall_result = {str(user_id): rec_items for user_id, rec_items in recall_result.items()}

        with open(file_path, 'w') as f:
            json.dump(json_recall_result, f)

        logger.info(f"召回结果已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存失败: {e}")






