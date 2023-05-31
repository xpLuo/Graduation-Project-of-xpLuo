import os
import pandas as pd

# the movielens 1m and 10m are process in similar ways
# the 'ratings.dat' file comes directly from the original dataset file downloaded from https://grouplens.org/datasets/movielens/
ratings = pd.read_csv(os.path.join('ratings.dat'),
                      sep='::', engine='python', encoding='latin-1',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

# Save into ratings.csv
ratings.to_csv('movielens_1m_ratings.csv',
               sep='\t',
               header=True,
               encoding='latin-1',
               columns=['user_id', 'item_id', 'rating'])
