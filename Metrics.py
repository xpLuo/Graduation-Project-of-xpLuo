import numpy as np


def calculate_average_probability_per_item_type(rating, predicted_probability_of_being_liked):
    """
    Introduction:
        We partition the items into “likable” and “not likable” based on whether their assigned score at time t,
        the rating of the ite is positive or negative. We measure the average probability for likable and not likable items.
    Args:
        rating (np.ndarray): the rating of every item for every user
        predicted_probability_of_being_liked (np.ndarray): the probability of every item being liked by every user
    Returns:
        likeable_average_probability (np.ndarray): the calculated result
        non_likeable_average_probability (np.ndarray): the calculated result
    """
    user_number, item_number = np.shape(predicted_probability_of_being_liked)

    likeable_average_probability = np.zeros(user_number)
    non_likeable_average_probability = np.zeros(user_number)

    for m in range(user_number):
        number_likeable, number_non_likeable = 0, 0

        for n in range(item_number):
            if rating[m, n] >= 0:
                number_likeable += 1
                likeable_average_probability[m] += predicted_probability_of_being_liked[m, n]
            elif rating[m, n] < 0:
                number_non_likeable += 1
                non_likeable_average_probability[m] += predicted_probability_of_being_liked[m, n]

        likeable_average_probability[m] = likeable_average_probability[m] / number_likeable
        non_likeable_average_probability[m] = non_likeable_average_probability[m] / number_non_likeable

    return likeable_average_probability, non_likeable_average_probability


def calculate_probability_mass_of_five_percent_selected(index, probability_of_being_selected):
    """
    Introduction:
        We consider the top 5% of items having the highest cosine similarity with initial user feature. These are the items
        that the system initially believes the user will like. We report the total probability mass of them being selected.
    Args:
        index (np.ndarray): the index of items having the highest cosine similarity with initial user feature
        probability_of_being_selected (np.ndarray): the probability of items being selected
    Returns:
        probability_mass (np.ndarray): the calculated result
    """
    user_number = len(probability_of_being_selected[:, 0])

    probability_mass = np.zeros(user_number)

    for i in range(user_number):
        for j in range(len(index[0])):
            probability_mass[i] = probability_mass[i] + probability_of_being_selected[i, index[i, j]]

    return probability_mass


def calculate_probability_mass_forged_item_selected(probability_of_being_selected):
    """
    Introduction:
        Consider the last hundred items that we forged, we have presented them repeatedly to the simulated user.
        And we report the total probability mass of them being selected as a metrics of how the recommender system perceive the change.
    Args:
        probability_of_being_selected (np.ndarray): the probability of items being selected
    Returns:
        probability_mass (np.ndarray): the calculated result
    """
    user_number, item_number = np.shape(probability_of_being_selected)

    probability_mass = np.zeros(user_number)

    for i in range(user_number):
        for j in range(100):
            probability_mass[i] = probability_mass[i] + probability_of_being_selected[i, item_number-j-1]

    return probability_mass


def calculate_average_probability_of_forged_items(rating, predicted_probability_of_being_liked):
    """
    Introduction:
        Consider the last hundred items that we forged, we have presented them repeatedly to the simulated user.
        And we report the average probability of the predicted probability of items being liked
        as a metrics of measuring the user's attitude towards this kind of items.
    Args:
        rating (np.ndarray): the rating of every item for every user
        predicted_probability_of_being_liked (np.ndarray): the probability of every item being liked by every user
    Returns:
        average_probability (np.ndarray): the calculated result
    """
    user_number, item_number = np.shape(predicted_probability_of_being_liked)

    rating_positive = np.zeros(user_number)
    rating_negative = np.zeros(user_number)

    for i in range(user_number):
        for j in range(100):
            if rating[i, item_number - j - 1] > 0:
                rating_positive[i] += 1
            else:
                rating_negative[i] += 1

    positive_index = []
    negative_index = []
    non_certain_index = []

    for i in range(user_number):
        if rating_positive[i] >= 90:
            positive_index.append(i)
        elif rating_negative[i] >= 90:
            negative_index.append(i)
        else:
            non_certain_index.append(i)

    non_certain_rating_positive = []
    non_certain_rating_negative = []

    if non_certain_index:
        for i in range(len(non_certain_index)):
            k = non_certain_index[i]
            print(k, ': ', rating_positive[k], ',', rating_negative[k])
            non_certain_rating_positive.append(rating_positive[k])
            non_certain_rating_negative.append(rating_negative[k])

    # positive_user_average_rating = np.zeros(len(positive_index))
    # negative_user_average_rating = np.zeros(len(negative_index))
    positive_user_average_probability = np.zeros(len(positive_index))
    negative_user_average_probability = np.zeros(len(negative_index))

    # for i in range(len(positive_index)):
    #     k = positive_index[i]
    #     for j in range(100):
    #         positive_user_average_rating[i] = positive_user_average_rating[i] + rating[k, item_number-j-1]
    #     positive_user_average_rating[i] = positive_user_average_rating[i] / 100
    # pr = positive_user_average_rating.mean()

    # for i in range(len(negative_index)):
    #     k = negative_index[i]
    #     for j in range(100):
    #         negative_user_average_rating[i] = negative_user_average_rating[i] + rating[k, item_number - j - 1]
    #     negative_user_average_rating[i] = negative_user_average_rating[i] / 100
    # nr = negative_user_average_rating.mean()

    for i in range(len(positive_index)):
        k = positive_index[i]
        for j in range(100):
            positive_user_average_probability[i] = positive_user_average_probability[i] + predicted_probability_of_being_liked[k, item_number-j-1]
        positive_user_average_probability[i] = positive_user_average_probability[i] / 100
    p = positive_user_average_probability.mean()

    for i in range(len(negative_index)):
        k = negative_index[i]
        for j in range(100):
            negative_user_average_probability[i] = negative_user_average_probability[i] + predicted_probability_of_being_liked[k, item_number - j - 1]
        negative_user_average_probability[i] = negative_user_average_probability[i] / 100
    n = negative_user_average_probability.mean()

    return p, n, non_certain_index, non_certain_rating_positive, non_certain_rating_negative
