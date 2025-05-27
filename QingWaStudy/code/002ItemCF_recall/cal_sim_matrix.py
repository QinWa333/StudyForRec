import math
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import  tqdm
import logging
import json
import os

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


def generate_user_item_dict(train_click_df):
    """
    生成用户点击电影的字典
    :param train_click_df:
    :return: user_item_dict
     {'uid1':[movie_id1,movie_id2],
     'uid2':[....]
     }
    """
    user_item_dict = {}
    # 按用户ID分组并遍历
    for user_id, group in train_click_df.groupby('user_id'):
        # 获取该用户点击的所有物品ID集合（去重，实际上数据中没有重复）
        item_list = group['movie_id'].tolist()
        # 添加到字典
        user_item_dict[user_id] = item_list

    return user_item_dict


def calculate_similarity(user_item_dict):
    """

    :param user_item_dict:
    :return: sim_dict
    sim_dict[item1][item2] = score
    """
    # 计算余弦相似度
    item_cnt = defaultdict(int)
    sim_dict = {}
    for uid, items in tqdm(user_item_dict.items()):
        for loc1,item in enumerate(items):
            item_cnt[item] +=1
            sim_dict.setdefault(item,{})

            for loc2,related_item in enumerate(items):
                if item == related_item:
                    continue

                sim_dict[item].setdefault(related_item,0)

                # 离得越近的两个物品越接近
                loc_weight = 0.7**(np.abs(loc2-loc1)-1)
                sim_dict[item][related_item] += loc_weight / (math.log(1+len(items)))
                #sim_dict[item][related_item] += loc_weight

    # 当前字典存的共现次数，再除以物品向量模长
    for item,related_items in tqdm(sim_dict.items()):
        for related_item,score in related_items.items():
            sim_dict[item][related_item] = score / (math.sqrt(item_cnt[item]*item_cnt[related_item]))
    return sim_dict


if __name__ == '__main__':
    logger.info('########## 1.考虑两个item再点击序列中的距离(0.7),越近相似度越高 2.对活跃用户进行打压（用户观看越多，他的序列中的电影相似度越低）' )
    save_path = '../../tmp_result/ItemCF'

    # 读取训练集点击记录
    train_click_df = pd.read_parquet('../../dataset/processed_df/train.parquet')
    user_item_dict = generate_user_item_dict(train_click_df)
    sim_dict = calculate_similarity(user_item_dict)

    # 将用户点击序列保存为json，当前是按时间顺序 旧->新
    file_path = os.path.join(save_path,'user_item_dict.json')
    with open(file_path, 'w')as f:
        json_user_item_dict = {k:list(v) for k,v in user_item_dict.items()}
        json.dump(json_user_item_dict,f)
    logger.info(f'用户交互序列已保存到{save_path}目录下')

    # 将相似度字典保存为json文件
    file_path = os.path.join(save_path,'item_similarity.json')
    with open(file_path, 'w') as f:
        # 将集合转换为列表
        json_sim_dict = {k: list(v.items()) for k, v in sim_dict.items()}
        json.dump(json_sim_dict, f)
    logger.info(f'相似度字典已保存到{save_path}目录下')