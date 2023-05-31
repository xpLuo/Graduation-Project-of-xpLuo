import sys
import numpy as np
import matplotlib.pyplot as plt
from InitialTraining import initial_training_real_data, synthetic_data_uniform, synthetic_data_mixture, forge_items
from Recommend import recommend
from Metrics import calculate_average_probability_per_item_type, calculate_probability_mass_of_five_percent_selected
from Utils import scaling_factor, drift_function, sigmoid


def preference_amplification_simulation(beta=1, gamma=0.4, eta=1, model_dimension=15, final_recommend_num=100, dataset='movielens 1m', method='kernel', improved=True,
                                        percentage=0.01, enlarge=False, reset_eta=0, step=20):
    """
    This function is the version of the simulation experiment contains merely preference amplification.
    This function essentially is a simpler/earlier version of 'main.py'
    It is used to draw the pictures that compares the different value for each parameter.
    The parameters in the first row are the function's input default value, only one parameter changes at one time to observe their effect,
    the parameters in the second row are the ones that should be able to mitigate the preference amplification effect.
    The function returns four vector used to plot the picture, size 'iteration'
    """
    # init parameters
    iteration = 200
    user_number = 50
    generate_item_number = 10000  # only used when generating synthetic data

    # init variables
    user_feature = np.zeros((user_number, model_dimension, iteration))
    likeable_average_probability = np.zeros((user_number, iteration))
    non_likeable_average_probability = np.zeros((user_number, iteration))
    probability_mass_of_top_five_percent_selected = np.zeros((user_number, iteration))

    # gain user feature and item feature via initial training
    if dataset == 'movielens 1m' or dataset == 'movielens 10m' or dataset == 'netflix':
        user_feature[:, :, 0], item_feature, global_mean, user_bias, item_bias = \
            initial_training_real_data(user_num=user_number, model_factor_num=model_dimension, verbose=0, dataset=dataset, method=method, seed=1)
        # the examination of method input correction can still be executed
    elif dataset == 'uniform':
        user_feature[:, :, 0], item_feature, global_mean, user_bias, item_bias = \
            synthetic_data_uniform(user_num=user_number, item_num=generate_item_number, model_factor_num=model_dimension, verbose=0)
    elif dataset == 'mixture':
        user_feature[:, :, 0], item_feature, global_mean, user_bias, item_bias = \
            synthetic_data_mixture(user_num=user_number, item_num=generate_item_number, model_factor_num=model_dimension, mix_factor=percentage, verbose=0)
    else:
        # the examination of dataset input in 'DataProcessing.py' will not be engaged since in this function the name has been tested
        print('ERROR, PLEASE SELECT A DATASET FROM THE FOLLOWING!!!')
        print('movielens 1m', ' \t ', 'movielens 10m', ' \t ', 'netflix', ' \t ', 'uniform', ' \t ', 'mixture')
        sys.exit(1)

    item_number = len(item_feature[0])
    true_probability_of_recommendation_being_liked = np.zeros((user_number, item_number))
    true_probability_of_recommendation_being_liked_last_time = np.zeros((user_number, item_number))

    # record the items that correspond best with users' initial feature vector
    rating_first_time = np.dot(user_feature[:, :, 0], item_feature)
    rating_predict_sort_index = np.argsort(rating_first_time, axis=1)
    index_of_top_five_percent_initial = rating_predict_sort_index[:, -round(0.05 * item_number):]
    if improved:
        for m in range(user_number):
            for n in range(item_number):
                true_probability_of_recommendation_being_liked_last_time[m, n] = sigmoid(rating_first_time[m, n])

    # interaction between user and recommender system and observe how user feature changes
    for k in range(1, iteration):
        # degenerative eta if applicable
        if k % step == 0:
            if reset_eta == 1:
                eta = 1 / (1 + k // step)
            elif reset_eta == 2:
                eta = 1 / (1 + np.sqrt(k // step))
            elif reset_eta == 3:
                eta = 1 * np.power(0.9, k // step)
        if reset_eta == 4:
            eta = 1 * np.exp(-0.9 / k)

        # enlarge the item base if applicable
        if enlarge:
            if k % step == 0:
                item_feature = forge_items(0, item_feature)
                item_number = len(item_feature[0])
                true_probability_of_recommendation_being_liked = np.zeros((user_number, item_number))
                true_probability_of_recommendation_being_liked_last_time = np.zeros((user_number, item_number))

        # generate recommendation
        rating, predicted_probability_of_being_liked, probability_of_being_selected, final_recommendation_index \
            = recommend(user_feature[:, :, k - 1], item_feature, global_mean, user_bias, item_bias, beta, k=final_recommend_num, verbose=0)

        # calculate average likeable and non-likeable predicted probability
        likeable_average_probability[:, k - 1], non_likeable_average_probability[:, k - 1] = \
            calculate_average_probability_per_item_type(rating, predicted_probability_of_being_liked)

        # calculate probability mass on items correlating well with the initial
        probability_mass_of_top_five_percent_selected[:, k - 1] = \
            calculate_probability_mass_of_five_percent_selected(index_of_top_five_percent_initial, probability_of_being_selected)

        # calculate 'pi(x)', i.e. the true probability of recommendation being liked by the user
        if improved:
            for m in range(user_number):
                for n in range(item_number):
                    true_probability_of_recommendation_being_liked[m, n] = \
                        true_probability_of_recommendation_being_liked_last_time[m, n] \
                        + gamma * scaling_factor(rating[m, n]) * drift_function(rating[m, n])
            true_probability_of_recommendation_being_liked_last_time = true_probability_of_recommendation_being_liked
        else:
            for m in range(user_number):
                for n in range(item_number):
                    true_probability_of_recommendation_being_liked[m, n] = \
                        predicted_probability_of_being_liked[m, n] \
                        + gamma * scaling_factor(rating[m, n]) * drift_function(rating[m, n])

        # calculate the iteration of user feature inside the recommender system
        for m in range(user_number):
            grads = np.zeros((model_dimension, final_recommend_num))
            for n in range(final_recommend_num):
                o = int(final_recommendation_index[m, n])
                grads[:, n] = item_feature[:, o] * probability_of_being_selected[m, o] * (
                        predicted_probability_of_being_liked[m, o] - true_probability_of_recommendation_being_liked[m, o])
            gradient = np.sum(grads, axis=1)
            user_feature[m, :, k] = user_feature[m, :, k - 1] - eta * gradient

        # the user feature generate by neural method almost definitely exceeds 1, thus does not require this
        # however the kernel method should restrict every element in the user feature within [-1,1]
        if method == 'kernel':
            for m in range(user_number):
                for n in range(model_dimension):
                    if user_feature[m, n, k] > 1:
                        user_feature[m, n, k] = 1
                    if user_feature[m, n, k] < -1:
                        user_feature[m, n, k] = -1

    # after the iteration, there is one last time that we need to calculate the metrics
    rating, predicted_probability_of_being_liked, probability_of_being_selected, final_recommendation_index \
        = recommend(user_feature[:, :, iteration - 1], item_feature, global_mean, user_bias, item_bias, beta, k=final_recommend_num, verbose=0)
    likeable_average_probability[:, iteration - 1], non_likeable_average_probability[:, iteration - 1] = \
        calculate_average_probability_per_item_type(rating, predicted_probability_of_being_liked)
    probability_mass_of_top_five_percent_selected[:, iteration - 1] = \
        calculate_probability_mass_of_five_percent_selected(index_of_top_five_percent_initial, probability_of_being_selected)

    # calculate user feature norm
    plot_user_feature_norm = np.zeros((user_number, iteration))
    for m in range(user_number):
        plot_user_feature = user_feature[m, :, :]
        for k in range(iteration):
            plot_user_feature_norm[m, k] = np.linalg.norm(plot_user_feature[:, k], ord=2)
    plot_all_user_feature_norm = np.mean(plot_user_feature_norm, axis=0)

    plot_all_user_likeable_average_probability = np.mean(likeable_average_probability, axis=0)
    plot_all_user_non_likeable_average_probability = np.mean(non_likeable_average_probability, axis=0)

    plot_all_user_probability_mass_of_top_five_percent_selected = np.mean(probability_mass_of_top_five_percent_selected, axis=0)

    return plot_all_user_feature_norm, plot_all_user_likeable_average_probability, plot_all_user_non_likeable_average_probability, \
        plot_all_user_probability_mass_of_top_five_percent_selected


if __name__ == '__main__':
    # Obtain data for the picture
    # change the code of the lines that are annotated to change the variable
    gl = [0.2, 0.4, 0.6, 0.8]  # gamma=gl[i]
    bl = [0.5, 1, 1.5, 2]  # beta=bl[i]
    el = [0.5, 1, 1.5, 2]  # eta=el[i]
    ml = [5, 15, 30, 60]  # model_dimension=ml[i]
    fl = [10, 50, 100, 150]  # final_recommend_num=fl[i]
    drl = ['movielens 1m', 'movielens 10m', 'netflix']  # dataset=drl[i]
    dsl = ['uniform', 'mixture']  # dataset=dsl[i]
    mel = ['kernel', 'neural']  # method=mel[i]
    il = [False, True]  # improved=il[i]
    pl = [0.01, 0.25, 0.5, 0.75]  # percentage=pl[i]
    # pl = [0.01, 0.05, 0.1, 0.15]
    rel = [0, 1, 2, 3, 4]  # reset_eta=rel[i]
    sl = [10, 20, 40]  # step=sl[i]

    s = sl  # !!Change!!
    num = len(s)

    p1 = np.zeros((200, num))
    p21 = np.zeros((200, num))
    p22 = np.zeros((200, num))
    p3 = np.zeros((200, num))

    # All the parameters of the function 'simulation'
    # beta = 1, gamma = 0.4, eta = 1, model_dimension = 15, final_recommend_num = 100, dataset = 'movielens 1m', method = 'kernel', improved = True,
    # percentage = 0.01, enlarge = False, reset_eta = 0, step = 20
    for i in range(num):
        p1[:, i], p21[:, i], p22[:, i], p3[:, i] = preference_amplification_simulation(dataset='movielens 10m', enlarge=True, step=s[i])  # !!Change!!
        print(i)

    # Plot
    # fig = plt.figure(figsize=(40, 9), dpi=300)
    plt.figure(figsize=(22, 15), dpi=300)
    plt.rcParams.update({'font.size': 21})

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.21, hspace=0.24)

    plt.subplot(2, 2, 1)
    for i in range(num):
        plt.plot(p1[:, i], label=str(s[i]))
    plt.legend(loc='best')
    plt.title('User Feature Norm')
    plt.xlabel('Step')
    plt.ylabel('Norm')

    plt.subplot(2, 2, 2)
    for i in range(num):
        plt.plot(p3[:, i], label=str(s[i]))
    plt.legend(loc='best')
    plt.title('Prob. of Well-correlation Items')
    plt.xlabel('Step')
    plt.ylabel('Prob. of System Select')

    plt.subplot(2, 2, 3)
    for i in range(num):
        plt.plot(p21[:, i], label=str(s[i]))
    plt.legend(loc='best')
    plt.title('Likeable Item Prob.')
    plt.xlabel('Step')
    plt.ylabel('Prob. of user liking')

    plt.subplot(2, 2, 4)
    for i in range(num):
        plt.plot(p22[:, i], label=str(s[i]))
    plt.legend(loc='best')
    plt.title('Non-likeable Item Prob.')
    plt.xlabel('Step')
    plt.ylabel('Prob. of user not liking')

    plt.suptitle('Effect of Enlarging the Item Base on Experiment Result')  # !!Change!!

    # only generate one legend in the main plot
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='lower center', ncols=num)

    plt.savefig('./pic/10m_enlarge', bbox_inches='tight')  # !!Change!!
    plt.show()
