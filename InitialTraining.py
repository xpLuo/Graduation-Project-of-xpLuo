import sys
import pandas as pd
from matrix_factorization import KernelMF
from DataProcessing import data_preparation
import numpy as np
from NeuralMFModel import build_network
from keras.callbacks import EarlyStopping, ModelCheckpoint


def synthetic_data_uniform(user_num: int = 50, item_num: int = 10000, model_factor_num: int = 15, verbose: int = 0):
    """
    Introduction:
        Generate synthetic data.
        The synthetic data is directly
        We initialize the user feature vector following the uniform distribution between [-0.3,0.3]
        Each item feature is generated independently following the uniform distribution between [-1,1]
    Args:
        user_num (int): the number of users
        item_num (int): the number of items
        model_factor_num (int): the number of how many dimension used in the matrix factorization to depict user and item feature
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
    Returns:
        user_feature_initial (np.ndarray): user feature, shape 'user_num * model_factor_num'
        item_feature_initial (np.ndarray): item feature, shape 'model_factor_num * item_num'
    """
    user_feature_initial = np.zeros((user_num, model_factor_num))
    item_feature_initial = np.zeros((model_factor_num, item_num))

    for i in range(user_num):
        user_feature_initial[i, :] = np.random.uniform(-0.3, 0.3, model_factor_num)

    for i in range(item_num):
        item_feature_initial[:, i] = np.random.uniform(-1, 1, model_factor_num)

    global_mean = np.random.randint(3, 4)
    user_bias = np.random.uniform(-0.5, 0.5, user_num)
    item_bias = np.random.uniform(-0.2, 0.2, item_num)

    if verbose == 1:
        print(user_feature_initial)
        print(item_feature_initial)
        print(np.min(user_feature_initial), np.max(user_feature_initial))
        print(np.min(item_feature_initial), np.max(item_feature_initial))
        print(np.shape(user_feature_initial), np.shape(item_feature_initial))

    return user_feature_initial, item_feature_initial, global_mean, user_bias, item_bias


def synthetic_data_mixture(user_num: int = 50, item_num: int = 10000, model_factor_num: int = 15, mix_factor: float = 0.01, verbose: int = 0):
    """
    Introduction:
        Generate synthetic data.
        We initialize the user feature vector following the uniform distribution between [-0.3,0.3]
        The item feature on the other hand, was generated following two types of distribution, one uniform between [-1,0.8], one uniform between [0.8,1]
        This was meant to mimic the situation that a special kind of items that have very similar features in a general environment
        and observe how the user may interact towards those special items which in real life may be problematic
    Args:
        user_num (int): the number of users
        item_num (int): the number of items
        model_factor_num (int): the number of how many dimension used in the matrix factorization to depict user and item feature
        mix_factor (float): the factor of how much percentage of
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
    Returns:
        user_feature_initial (np.ndarray): user feature, shape 'user_num * model_factor_num'
        item_feature_initial (np.ndarray): item feature, shape 'model_factor_num * item_num'
    """
    user_feature_initial = np.zeros((user_num, model_factor_num))
    item_feature_initial = np.zeros((model_factor_num, item_num))

    for i in range(user_num):
        user_feature_initial[i, :] = np.random.uniform(-0.3, 0.3, model_factor_num)

    for i in range(item_num):
        item_feature_initial[:, i] = np.random.uniform(-1, 0.8, model_factor_num)

    # choose 1% of the item and regenerate their item feature following another distribution
    select_num = int(mix_factor * item_num)
    index = np.arange(item_num)
    select_index = np.random.choice(index, select_num, replace=False)

    for i in range(select_num):
        item_feature_initial[:, select_index[i]] = np.random.uniform(0.8, 1, model_factor_num)

    global_mean = np.random.randint(3, 4)
    user_bias = np.random.uniform(-0.5, 0.5, user_num)
    item_bias = np.random.uniform(-0.2, 0.2, item_num)

    if verbose == 1:
        print(user_feature_initial)
        print(item_feature_initial)
        print(np.min(user_feature_initial), np.max(item_feature_initial))
        print(np.shape(user_feature_initial), np.shape(item_feature_initial))

    return user_feature_initial, item_feature_initial, global_mean, user_bias, item_bias


def initial_training_real_data(user_num: int = 50, model_factor_num: int = 15, verbose: int = 0,
                               dataset: str = '1m', method: str = 'kernel', seed: int = 0):
    """
    Introduction:
        Conduct initial training on real world dataset in attempt to obtain user feature and item feature
    Args:
        user_num (int): the number of how many users included in the matrix factorization
        model_factor_num (int): the number of how many dimension used in the matrix factorization to depict user and item feature
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
        dataset (str): the string to determine whether load the data from movielens 1m or movielens 10m
        method (str): the string to determine which method of matrix factorization to be used, kernel or neural
        seed (int): the seed of random state when shuffle the data
    Returns:
        user_feature_initial (np.ndarray): user feature, shape 'user_num * model_factor_num'
        item_feature_initial (np.ndarray): item feature, shape 'model_factor_num * item_num'
    """
    # gain data from dataset movielens
    x_train, y_train = data_preparation(user_num, verbose, dataset)

    item_num = x_train['item_id'].drop_duplicates().nunique()
    user_feature_initial = np.zeros((user_num, model_factor_num))
    item_feature_initial = np.zeros((model_factor_num, item_num))
    global_mean_initial = y_train.mean()
    user_bias_initial = np.zeros(user_num)
    item_bias_initial = np.zeros(item_num)

    # initial training using method 'Matrix Factorization'

    # kernel MF
    if method == 'kernel':
        # kernel matrix factorization
        matrix = KernelMF(n_epochs=20, n_factors=model_factor_num, verbose=0, lr=0.001, reg=0.005)

        # shuffle the data randomly in order to gain better performance
        shuffled_x = x_train.sample(frac=1., random_state=seed)
        shuffled_y = y_train.sample(frac=1., random_state=seed)

        # train the model
        matrix.fit(shuffled_x, shuffled_y)
        # gain the user feature and item feature that are required for later interaction
        user_feature_initial, item_feature_initial = matrix.user_features, matrix.item_features
        # do a transpose so that 'user feature' and 'item feature' can do dot product
        item_feature_initial = item_feature_initial.transpose()

        # introduce user biases and item biases
        user_bias_initial = matrix.user_biases
        item_bias_initial = matrix.item_biases

        # output a few variables for debugging
        if verbose == 1:
            print(user_feature_initial, '\n', item_feature_initial, '\n')
            print(np.shape(user_feature_initial), np.shape(item_feature_initial), '\n')
            print(user_bias_initial, '\n', item_bias_initial)
            print(np.shape(user_bias_initial), np.shape(item_bias_initial))

    # neural MF
    if method == 'neural':
        y_train = y_train - global_mean_initial
        # concat the data in order for them to be easier to shuffle
        all_data = pd.concat([x_train, y_train], axis=1)
        # shuffle the data randomly in order to gain better performance
        shuffled_data = all_data.sample(frac=1., random_state=seed)
        # transfer data from pd.dataframe/pd.series into numpy.ndarray so that they can be accepted as input to keras
        user = shuffled_data['user_id'].values
        item = shuffled_data['item_id'].values
        rating = shuffled_data['rating'].values

        # the purpose of rearranging the id is to make the training more effective
        # rearrange the id of the users
        temp = np.unique(user)
        # the first step would be creating a mapping relation between the original id and the new id and here we use a 'dictionary' to represent
        mapping = {}
        for i in range(len(temp)):
            mapping[i] = temp[i]
        # take out the keys and values of the dictionary independently
        # this step further creates a relation between the keys and an arithmetic progression starting from 0
        # as well as the relation between the values and the same arithmetic progression
        k = np.array(list(mapping.keys()))
        v = np.array(list(mapping.values()))
        # this mapping_ar is actually swap the key and the value of the previous dictionary
        # since its key is new id we want and the value is the old id we want to replace
        # making it np.ndarray also enables the automatic assignment
        mapping_ar = np.zeros(max(v) + 1, dtype=v.dtype)
        mapping_ar[v] = k
        # this is the final step of changing every id according to the map
        out_user = mapping_ar[user]

        # rearrange the id of the items
        temp = np.unique(item)
        mapping = {}
        for i in range(len(temp)):
            mapping[i] = temp[i]
        k = np.array(list(mapping.keys()))
        v = np.array(list(mapping.values()))
        mapping_ar = np.zeros(max(v) + 1, dtype=v.dtype)
        mapping_ar[v] = k
        out_item = mapping_ar[item]

        # neural network
        model = build_network(user_num, item_num, model_factor_num)
        callbacks = [EarlyStopping('val_loss', patience=5),
                     ModelCheckpoint('weights.h5', save_best_only=True)]
        # train the model
        # model.fit([user, item], rating, epochs=50, validation_split=0.1, callbacks=callbacks)
        model.fit([out_user, out_item], rating, epochs=50, validation_split=0.1, callbacks=callbacks)

        # the feature matrix is indeed the weight matrix of the embedding layer
        layer_user = model.get_layer('user_embedding')
        layer_item = model.get_layer('item_embedding')
        user_feature_initial = np.array(layer_user.get_weights())[0]
        item_feature_initial = np.array(layer_item.get_weights())[0]
        item_feature_initial = item_feature_initial.transpose()

        # introduce user biases and item biases
        layer_user_biases = model.get_layer('user_bias')
        layer_item_biases = model.get_layer('item_bias')
        user_bias_initial = np.array(layer_user_biases.get_weights())[0]
        item_bias_initial = np.array(layer_item_biases.get_weights())[0]
        user_bias_initial = user_bias_initial.reshape(-1)
        item_bias_initial = item_bias_initial.reshape(-1)

        if verbose == 1:
            print(x_train, '\n', y_train, '\n', all_data, '\n')
            print(user_feature_initial, '\n', item_feature_initial, '\n')
            print(type(user_feature_initial), np.shape(user_feature_initial))
            print(user_bias_initial, '\n', item_bias_initial)
            print(np.shape(user_bias_initial), np.shape(item_bias_initial))

    # method input examination
    if user_feature_initial.any() == 0:
        print('ERROR, PLEASE SELECT A TRAINING METHOD FROM THE FOLLOWING!!!')
        print('kernel', ' \t ', 'neural')
        sys.exit(1)

    return user_feature_initial, item_feature_initial, global_mean_initial, user_bias_initial, item_bias_initial


def forge_items(start, item_feature, verbose: int = 0):
    """
    Introduction:
        In this function, we forged a group of items that are very similar to each other and will try to recommend them as frequent as possible
    Args:
        start (float): if start==1, then enter user induction and generate a set of items that are similar to each other;
                       if start==0, then enter item base enlargement and generate a set of items that includes 10 genres
        item_feature (np.ndarray): the item feature matrix, shape 'model_factor_num * item_num'
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
    Returns:
        item_feature (np.ndarray): the new item feature matrix, which now contains 100 more columns of items feature vector
    """
    model_factor_num = len(item_feature[:, 0])
    item_feature_forged = np.zeros((model_factor_num, 100))
    if start == 1:
        # this case is user induction
        np.random.seed(1)
        item_feature_forged_median = np.random.uniform(-1, 1, model_factor_num)
        for i in range(100):
            item_feature_forged[:, i] = item_feature_forged_median + np.random.uniform(-0.05, 0.05, model_factor_num)
        print('Forge item feature median: \n', item_feature_forged_median)
    elif start == 0:
        # this case is item base enlargement
        for k in range(10):
            item_feature_forged_median = np.random.uniform(-0.95, 0.95, model_factor_num)
            for i in range(10):
                item_feature_forged[:, i] = item_feature_forged_median + np.random.uniform(-0.1, 0.1, model_factor_num)

    for i in range(model_factor_num):
        for j in range(100):
            if item_feature_forged[i, j] > 1:
                item_feature_forged[i, j] = 1

    item_feature = np.concatenate((item_feature, item_feature_forged), axis=1)

    if verbose == 1:
        print(item_feature_forged)
        print(item_feature)
        print(np.shape(item_feature))

    return item_feature


if __name__ == '__main__':
    # synthetic_data_uniform(verbose=1)
    # synthetic_data_mixture(verbose=1)
    u_f, i_f, m, u_b, i_b = initial_training_real_data(verbose=0, dataset='movielens 1m', method='neural')
    forge_items(0.9, i_f)
