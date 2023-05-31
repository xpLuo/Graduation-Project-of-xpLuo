import math
import numpy as np
from Utils import sigmoid
from InitialTraining import initial_training_real_data


def recommend(user_feature, item_feature, global_mean, user_bias, item_bias, beta: float = 1, k: int = 100, verbose: int = 0):
    """
    Introduction:
        Use the received user feature and item feature to recommend items to user
        First the method will rate every movie for every user
        Then select one movie from the top K movies for them using the idea of softmax
    Args:
        user_feature (np.ndarray): user feature
        item_feature (np.ndarray): item feature
        global_mean (float): the mean of all ratings
        user_bias (np.ndarray): user bias
        item_bias (np.ndarray): item bias
        beta (float): system sensitivity, determines the steepness of 'ft', high beta cause only the highest score get recommended
        k (int): the number of items that are finally selected to present to the user according the probability of being selected
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
    Returns:
        rating_predict (np.ndarray): the dot product of user feature and item feature
        predicted_probability_being_liked (np.ndarray): the probability of selected item to be liked by the user predicted by the recommender system
                                                        actually to be the sigmoid function of predicted rating
        probability_being_selected (np.ndarray): the probability of the selected item being finally recommended by the recommender system
                                                 actually to be the softmax function of predicted rating
    """
    user_number, feature_number = np.shape(user_feature)
    feature_number, item_number = np.shape(item_feature)

    # calculate the rating, 'user' in row, 'item' in column
    rating_predict = np.dot(user_feature, item_feature)

    # add the global mean, user bias and item bias
    # NOT APPLICABLE
    # for i in range(user_number):
    #     for j in range(item_number):
    #         rating_predict[i, j] = (global_mean + user_bias[i] + item_bias[j] + 5 * rating_predict[i, j]) / 5

    # calculate the probability of selected item to be liked by the user predicted by the recommender system
    predicted_probability_being_liked = np.zeros((user_number, item_number))
    for i in range(user_number):
        for j in range(item_number):
            predicted_probability_being_liked[i, j] = sigmoid(rating_predict[i, j])

    # calculate the probability of the selected item being finally recommended by the recommender system
    probability_being_selected = np.zeros((user_number, item_number))
    for i in range(user_number):
        for j in range(item_number):
            probability_being_selected[i, j] = math.exp(beta * rating_predict[i, j])
    row_sum = probability_being_selected.sum(axis=1)
    for j in range(item_number):
        probability_being_selected[:, j] = probability_being_selected[:, j] / row_sum

    # determine a final recommendation
    final_recommendation = np.zeros((user_number, k))
    for i in range(user_number):
        # this function actually returns the index of the final recommended item for each user
        # since all items are being considered, even the items that predicted disliked may be recommended
        final_recommendation[i, :] = np.random.choice(a=np.arange(item_number), size=k, p=probability_being_selected[i, :])

    # output a few variables for debugging
    if verbose == 1:
        print(rating_predict, '\n', type(rating_predict), '\n', np.shape(rating_predict), '\n')
        print(np.min(rating_predict), np.max(rating_predict))
        print(predicted_probability_being_liked, '\n')
        print(probability_being_selected, '\n')
        print(final_recommendation, '\n')

    return rating_predict, predicted_probability_being_liked, probability_being_selected, final_recommendation


def induction(item_number, change_num, recommend_num, recommendation_index, verbose: int = 0):
    """
    Introduction:

    Args:
        item_number (int):
        change_num (int):
        recommend_num (int):
        recommendation_index (np.ndarray):
        verbose (int): whether to output a few variables for the purpose of debugging, 0 as no, 1 as yes
    Returns:
        recommendation_index (np.ndarray):
    """
    user_num = len(recommendation_index[:, 0])
    # ar1 is 1 to 100
    ar1 = np.arange(recommend_num)
    # ar2 is 3400 to 3499
    ar2 = np.linspace(item_number-100, item_number-1, 100, dtype=int)
    # choose the index in recommendation index and choose the index of item feature that will replace the original index
    change_index = np.zeros((user_num, change_num), dtype=int)
    for i in range(user_num):
        change_index[i, :] = np.random.choice(a=ar1, size=change_num, replace=False)
    change_content = np.random.choice(a=ar2, size=change_num)

    for i in range(user_num):
        for j in range(change_num):
            recommendation_index[i, change_index[i, j]] = change_content[j]

    if verbose == 1:
        print(change_index)
        print(change_content)
        print(recommendation_index)

    return recommendation_index


if __name__ == '__main__':
    u_f, i_f, m, u_b, i_b = initial_training_real_data(user_num=50, model_factor_num=15, verbose=0, dataset='movielens 1m', method='kernel', seed=1)
    rating, predicted_probability_of_being_liked, probability_of_being_selected, final_recommendation_index = recommend(u_f, i_f, m, u_b, i_b)
    induction(len(i_b)+100, 2, 100, final_recommendation_index, verbose=1)
