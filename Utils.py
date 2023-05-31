import math


def sigmoid(x: float) -> float:
    """
    Introduction:
        Calculate the sigmoid function of input, which primarily used in P = sigmoid(S) to
        calculate the probability of how much the recommender system believe the user may like the recommendation
    Args:
        x (float): Input, in fact as the rating of an item for a user
    Returns:
        y (float): Output, in fact as the predicted probability of item being liked by the user in the recommender system
    """
    y = 1 / (1 + math.exp(-x))
    return y


def scaling_factor(x: float) -> float:
    """
    Introduction:
        Calculate the scaling factor 'alpha', which is primarily used in calculating the true probability of recommendation
    Args:
        x (float): Input, the rating, which is also in fact the condition of this function
    Returns:
        y (float): Output
    """
    if x >= 0:
        y = 1 - sigmoid(x)
    else:
        y = sigmoid(x)
    return y


def drift_function(x: float) -> float:
    """
     Introduction:
        Calculate the drift function, which is primarily used in calculating the true probability of recommendation
    Args:
        x (float): Input, the rating
    Returns:
        y (float): Output
    """
    y = 2.33 * x * math.exp(- x ** 2)
    return y

