import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import pyarrow as pa
# 配置log日志
logging.basicConfig(
    filename='../log/test.log',
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

# 电影信息
mnames = ['movie_id', 'title', 'genres']
movies_df = pd.read_csv('../dataset/ml-1m/movies.dat',
                        sep='::',
                        header=None,
                        names=mnames,
                        engine='python',
                        encoding='ISO-8859-1')
# 评分信息
rnames = ['user_id', 'movie_id', 'score','timestamp']
ratings_df = pd.read_csv('../dataset/ml-1m/ratings.dat',
                         sep='::',
                         header=None,
                         engine='python',
                         names=rnames)

# 统计每个用户对每部电影的评分次数
user_movie_counts = ratings_df.groupby(['user_id', 'movie_id']).size().reset_index(name='count')

# 筛选出评分次数大于1的记录（即重复观看）
repeat_viewings = user_movie_counts[user_movie_counts['count'] > 1]

# 计算重复观看的用户比例
users_with_repeats = repeat_viewings['user_id'].nunique()
total_users = ratings_df['user_id'].nunique()
repeat_user_ratio = users_with_repeats / total_users * 100

# 计算重复观看的电影比例
movies_with_repeats = repeat_viewings['movie_id'].nunique()
total_movies = ratings_df['movie_id'].nunique()
repeat_movie_ratio = movies_with_repeats / total_movies * 100

# 输出结果
logger.info(f"共有 {len(repeat_viewings)} 条用户重复观看记录")
logger.info(f"{users_with_repeats} 个用户({repeat_user_ratio:.2f}%)有重复观看行为")
logger.info(f"{movies_with_repeats} 部电影({repeat_movie_ratio:.2f}%)被重复观看")

# 查看重复次数最多的用户和电影
top_users = repeat_viewings.groupby('user_id')['count'].sum().sort_values(ascending=False).head(5)
top_movies = repeat_viewings.groupby('movie_id')['count'].sum().sort_values(ascending=False).head(5)

logger.info("\n重复观看次数最多的用户:")
for user_id, count in top_users.items():
    logger.info(f"用户 {user_id}: {count} 次重复观看")

logger.info("\n重复观看次数最多的电影:")
for movie_id, count in top_movies.items():
    movie_title = movies_df.loc[movies_df['movie_id'] == movie_id, 'title'].iloc[0]
    logger.info(f"电影 {movie_id} ({movie_title}): {count} 次被重复观看")

