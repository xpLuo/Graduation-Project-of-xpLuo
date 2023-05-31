import sys
import pandas as pd


def data_preparation(user_num: int = 50, verbose: int = 0, dataset: str = '1m'):
    """
    Introduction:
        Prepare the dataset MovieLens 1M or 10M or netflix and pick out the users who rated most
    Args:
        user_num (int): the number of how many users included in the matrix factorization
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
        dataset (str): the string to determine whether to load the data from movielens 1m or movielens 10m or netflix
    Returns:
        x_train (pd.dataframe): data made up with the information of user id and item id
        y_train (pd.dataframe): data of rating made by user to item
    """
    # Movielens data found here https://grouplens.org/datasets/movielens/
    # Netflix data found here https://drive.google.com/drive/folders/1gRss2XCEi94xhEcH96Bf_KYV0SciF_iz?usp=sharing
    # PS: the movielens 1m or 10m file has already been preprocessed via the file 'DataPreProcessing' which has been proven to be unnecessary later
    # Reading ratings file
    if dataset == 'movielens 1m':
        # contains 100,0208 ratings
        ratings = pd.read_csv('movielens_1m_ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'item_id', 'rating'])
    elif dataset == 'movielens 10m':
        # contains 1000,0054 ratings
        ratings = pd.read_csv('movielens_10m_ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'item_id', 'rating'])
    elif dataset == 'netflix':
        # due to the large size of the original netflix dataset, here we use a subset but still contains 689,7746 ratings
        ratings = pd.read_csv('netflix_subset_ratings.txt', sep=' ', encoding='latin-1', names=['user_id', 'item_id', 'rating', 'date'])
        ratings = ratings.drop('date', axis=1)
    else:
        print('ERROR, PLEASE SELECT A DATASET FROM THE FOLLOWING!!!')
        print('movielens 1m', ' \t ', 'movielens 10m', ' \t ', 'netflix')
        sys.exit(1)
    # 6040 users, 3706 movies, rating 1 to 5 in movielens 1m  dataset
    # 69878 users, 10677 movies, rating 1 to 5 in movielens 10m dataset
    # 10000 users, 10000 movies, rating 1 to 5 in netflix dataset

    # choose 50 users who rated most
    # leaving 3400 movies in movielens 1m, 10208 movies in movielens 10m, 9989 movies in netflix
    # leaving 59640 ratings in movielens 1m, 153565 ratings in movielens 10m, 166173 ratings in netflix
    users_count = ratings.user_id.value_counts().head(user_num)
    # users_count data type: pd.series, which stores the number of a certain user's rating
    # thus the user id of the user who actually rated most is its index
    users_rated_most = users_count.index
    # select these users from the 'ratings' matrix and dump the rest
    ratings = ratings.query("user_id in @users_rated_most")

    x_train = ratings[["user_id", "item_id"]]
    y_train = ratings["rating"]

    # output a few variables for debugging
    if verbose == 1:
        print(ratings.nunique())
        print(x_train)
        print(y_train)
        print(users_count, '\n')
        print(users_rated_most, '\n')
        print(ratings.user_id.unique(), '\n')

    return x_train, y_train


if __name__ == '__main__':
    data_preparation(verbose=1, dataset='movielens 10m')
