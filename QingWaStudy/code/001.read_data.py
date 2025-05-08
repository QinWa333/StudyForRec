import numpy as np
import pandas as pd

# 用户
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
user_df = pd.read_csv('../dataset/ml-1m/users.dat',
                      sep='::',
                      header=None,
                      names=unames,
                      engine='python')
print(user_df)
# 电影信息
mnames = ['movie_id', 'title', 'genres']
movies_df = pd.read_csv('../dataset/ml-1m/movies.dat',
                        sep='::',
                        header=None,
                        names=mnames,
                        engine='python',
                        encoding='ISO-8859-1')
print(movies_df)
# 评分信息
rnames = ['user_id', 'movie_id', 'score','timestamp']
ratings_df = pd.read_csv('../dataset/ml-1m/ratings.dat',
                         sep='::',
                         header=None,
                         engine='python',
                         names=rnames)
print(ratings_df)