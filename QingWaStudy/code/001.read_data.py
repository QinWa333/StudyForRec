import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import pyarrow as pa
# 配置log日志
logging.basicConfig(
    filename='../log/001read_dada.log',
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


# 使用Parquet格式保存数据（高效IO）
def save_dataframe(df, path, compression='snappy'):
    try:
        df.to_parquet(path, compression=compression)
        logger.info(f"数据已保存至 {path} (格式: Parquet, 压缩: {compression})")
    except Exception as e:
        logger.error(f"保存数据失败: {str(e)}")
        raise

# 用户
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
user_df = pd.read_csv('../dataset/ml-1m/users.dat',
                      sep='::',
                      header=None,
                      names=unames,
                      engine='python')

logger.info("用户数据前几行：\n%s", user_df.head())
# 电影信息
mnames = ['movie_id', 'title', 'genres']
movies_df = pd.read_csv('../dataset/ml-1m/movies.dat',
                        sep='::',
                        header=None,
                        names=mnames,
                        engine='python',
                        encoding='ISO-8859-1')
logger.info("电影数据前几行：\n%s", movies_df.head())
# 评分信息
rnames = ['user_id', 'movie_id', 'score','timestamp']
ratings_df = pd.read_csv('../dataset/ml-1m/ratings.dat',
                         sep='::',
                         header=None,
                         engine='python',
                         names=rnames)
logger.info("评分数据前几行：\n%s", ratings_df.head())


# 数据处理
## 1.将电影名称后面的年份 单独作为1列
movies_df['create_year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
movies_df['title'] = movies_df['title'].str.replace(r' \(\d{4}\)$', '', regex=True)
# print(movies_df.head())
## 2.用户性别用01进行编码
user_df['gender'] = user_df['gender'].map({'F':0, 'M':1})
# print(user_df.head())


# 数据划分
# 随机取2000 用户作为验证集用户，其余用户作为测试集用户。
# 用户的最后一次观看电影作为验证集或测试集，奇遇观看记录作为训练集
all_users = user_df['user_id'].unique()
random_state = 666
test_users, val_users = train_test_split(
    all_users,
    test_size=2000,  # 直接指定验证集大小（注意：test_size需≤总用户数）
    random_state=random_state,
    shuffle=True  # 打乱顺序后抽样
)
print(val_users.size)# 2000
print(test_users.size)# 4040

# 划分训练集、验证集、测试集
ratings_df = ratings_df.sort_values('timestamp')
all_last_click = ratings_df.groupby('user_id').tail(1)
val_last_click_df = all_last_click[all_last_click['user_id'].isin(val_users)]
test_last_click_df = all_last_click[all_last_click['user_id'].isin(test_users)]
train_click_df = ratings_df[~ratings_df.index.isin(all_last_click.index)]

logger.info(f"验证集大小: {len(val_last_click_df)}, 用户数: {val_last_click_df['user_id'].nunique()}")
logger.info(f"测试集大小: {len(test_last_click_df)}, 用户数: {test_last_click_df['user_id'].nunique()}")
logger.info(f"训练集大小: {len(train_click_df)}, 用户数: {train_click_df['user_id'].nunique()}")

# 保存到文件（这里我尝试使用Parquet，据说比之间存csv快，顺便学习一下
# 读取使用 user_df = pd.read_parquet('../dataset/ml-1m/processed/users.parquet')
# 保存用户表和电影表
save_dataframe(user_df, '../dataset/processed_df/users.parquet')
save_dataframe(movies_df, '../dataset/processed_df/movies.parquet')

# 保存训练集、验证集、测试集
save_dataframe(train_click_df, '../dataset/processed_df/train.parquet')
save_dataframe(val_last_click_df, '../dataset/processed_df/validation.parquet')
save_dataframe(test_last_click_df, '../dataset/processed_df/test.parquet')