import pandas as pd
import numpy as np
import tqdm
import logging



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
     {'uid1':{movie_id1,movie_id2},
     'uid2':{....}
     }
    """
    user_item_dict = {}
    # 按用户ID分组并遍历
    for user_id, group in train_click_df.groupby('user_id'):
        # 获取该用户点击的所有物品ID集合（去重）
        item_set = set(group['movie_id'].tolist())
        # 添加到字典
        user_item_dict[user_id] = item_set

    return user_item_dict


def calculate_similarity(user_item_dict):


    return


if __name__ == '__main__':

    # 读取训练集点击记录
    train_click_df = pd.read_parquet('../../dataset/processed_df/train.parquet')
    user_item_dict = generate_user_item_dict(train_click_df)
    print(user_item_dict)