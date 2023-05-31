import sys
import numpy as np
import matplotlib.pyplot as plt
from InitialTraining import initial_training_real_data, synthetic_data_uniform, synthetic_data_mixture, forge_items
from Recommend import recommend, induction
from Metrics import calculate_average_probability_per_item_type, calculate_probability_mass_of_five_percent_selected, \
    calculate_probability_mass_forged_item_selected, calculate_average_probability_of_forged_items
from Utils import scaling_factor, drift_function, sigmoid


def user_induction_simulation(beta=1, gamma=0.4, eta=1, model_dimension=15, final_recommend_num=100, dataset='movielens 1m', method='kernel',
                              improved=True, replace_num=2):
    """
    This function is the version of the simulation experiment contains merely user induction.
    This file is essentially the 'main.py' that constantly set the variable 'induce' to be True.
    It is also used to draw the picture that can help us understand how user has behaved.
    """
    # init parameters
    iteration = 500
    user_number = 50
    generate_item_number = 10000  # only used when generating synthetic data

    # init variables
    user_feature = np.zeros((user_number, model_dimension, iteration))
    likeable_average_probability = np.zeros((user_number, iteration))
    non_likeable_average_probability = np.zeros((user_number, iteration))
    probability_mass_of_top_five_percent_selected = np.zeros((user_number, iteration))
    probability_mass_of_forged_item_selected = np.zeros((user_number, iteration))
    like_forged_items_average_probability = np.zeros(iteration)
    non_like_forged_items_average_probability = np.zeros(iteration)
    past_memory_of_non_certain_user = []
    all_non_certain_user = []
    start = []
    end = []
    ncu_rating_p = np.zeros((1, iteration))
    ncu_rating_n = np.zeros((1, iteration))

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
            synthetic_data_mixture(user_num=user_number, item_num=generate_item_number, model_factor_num=model_dimension, mix_factor=0.01, verbose=0)
    else:
        # the examination of dataset input in 'DataProcessing.py' will not be engaged since in this function the name has been tested
        print('ERROR, PLEASE SELECT A DATASET FROM THE FOLLOWING!!!')
        print('movielens 1m', ' \t ', 'movielens 10m', ' \t ', 'netflix', ' \t ', 'uniform', ' \t ', 'mixture')
        sys.exit(1)

    # add the items that we plan to induce the users
    item_feature = forge_items(1, item_feature)

    # init the variables that require item number
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
        # generate recommendation
        rating, predicted_probability_of_being_liked, probability_of_being_selected, final_recommendation_index \
            = recommend(user_feature[:, :, k - 1], item_feature, global_mean, user_bias, item_bias, beta, k=final_recommend_num, verbose=0)
        final_recommendation_index = induction(item_number, replace_num, final_recommend_num, final_recommendation_index)

        # calculate average likeable and non-likeable predicted probability
        likeable_average_probability[:, k - 1], non_likeable_average_probability[:, k - 1] = \
            calculate_average_probability_per_item_type(rating, predicted_probability_of_being_liked)

        # calculate probability mass on items correlating well with the initial
        probability_mass_of_top_five_percent_selected[:, k - 1] = \
            calculate_probability_mass_of_five_percent_selected(index_of_top_five_percent_initial, probability_of_being_selected)

        # calculate the probability of forged items of being selected
        probability_mass_of_forged_item_selected[:, k - 1] = \
            calculate_probability_mass_forged_item_selected(probability_of_being_selected)

        # calculate the average of the predicted probability of forged items of being liked
        # like_forged_items_average_rating[k - 1], non_like_forged_items_average_rating[k - 1], \
        like_forged_items_average_probability[k - 1], non_like_forged_items_average_probability[k - 1], \
            non_certain_user, non_certain_user_positive_rating_num, non_certain_user_negative_rating_num = \
            calculate_average_probability_of_forged_items(rating, predicted_probability_of_being_liked)

        # if there are any user who doesn't show a definite attitude, i.e. like or dislike, towards the forged items
        # then we are interested in how their attitude changes and thus record this
        if non_certain_user == past_memory_of_non_certain_user:
            pass
        else:
            # if a user leaves our watch list, then show us his/her ultimate attitude
            if len(non_certain_user) < len(past_memory_of_non_certain_user):
                past_user = set(non_certain_user) ^ set(past_memory_of_non_certain_user)
                past_user = list(past_user)
                for m in range(len(past_user)):
                    index_user = all_non_certain_user.index(past_user[m])
                    end[index_user] = k - 1
                    if ncu_rating_p[index_user, k - 2] < ncu_rating_n[index_user, k - 2]:
                        print('User ', past_user, ' has developed negative attitude towards forged item.')
                    else:
                        print('User ', past_user, ' has developed positive attitude towards forged item.')
            # if a user shows up on our watch list, no matter at the beginning or in the middle,
            # we should add he/she to our list of names and the array of recording
            if len(non_certain_user) > len(past_memory_of_non_certain_user):
                new_user = set(non_certain_user) ^ set(past_memory_of_non_certain_user)
                new_user = list(new_user)
                print('User', new_user, 'hold uncertain attitude towards forged item.')
                for m in range(len(new_user)):
                    all_non_certain_user.append(new_user[m])
                    start.append(k - 1)
                    end.append(k - 1)
                l1 = len(ncu_rating_p[:, 0])
                l2 = len(all_non_certain_user)
                for m in range(l2 - l1):
                    temp = np.zeros((1, iteration))
                    ncu_rating_p = np.vstack((ncu_rating_p, temp))
                    ncu_rating_n = np.vstack((ncu_rating_n, temp))
        # after making changes according to the watch list, time to add the new values
        for m in range(len(non_certain_user)):
            index_user = all_non_certain_user.index(non_certain_user[m])
            ncu_rating_p[index_user, k - 1] = non_certain_user_positive_rating_num[m]
            ncu_rating_n[index_user, k - 1] = non_certain_user_negative_rating_num[m]
        past_memory_of_non_certain_user = non_certain_user

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

        print(k)

    # after the iteration, there is one last time that we need to calculate the metrics
    rating, predicted_probability_of_being_liked, probability_of_being_selected, final_recommendation_index \
        = recommend(user_feature[:, :, iteration - 1], item_feature, global_mean, user_bias, item_bias, beta, k=final_recommend_num, verbose=0)
    likeable_average_probability[:, iteration - 1], non_likeable_average_probability[:, iteration - 1] = \
        calculate_average_probability_per_item_type(rating, predicted_probability_of_being_liked)
    probability_mass_of_top_five_percent_selected[:, iteration - 1] = \
        calculate_probability_mass_of_five_percent_selected(index_of_top_five_percent_initial, probability_of_being_selected)
    probability_mass_of_forged_item_selected[:, iteration - 1] = \
        calculate_probability_mass_forged_item_selected(probability_of_being_selected)
    # like_forged_items_average_rating[iteration - 1], non_like_forged_items_average_rating[iteration - 1], \
    like_forged_items_average_probability[iteration - 1], non_like_forged_items_average_probability[iteration - 1], \
        non_certain_user, non_certain_user_positive_rating_num, non_certain_user_negative_rating_num = \
        calculate_average_probability_of_forged_items(rating, predicted_probability_of_being_liked)

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
    plot_all_user_probability_mass_of_forged_item_selected = np.mean(probability_mass_of_forged_item_selected, axis=0)

    return plot_all_user_feature_norm, plot_all_user_likeable_average_probability, plot_all_user_non_likeable_average_probability, \
        plot_all_user_probability_mass_of_top_five_percent_selected, plot_all_user_probability_mass_of_forged_item_selected, \
        like_forged_items_average_probability, non_like_forged_items_average_probability, all_non_certain_user, start, end, ncu_rating_p, ncu_rating_n


if __name__ == '__main__':
    # Obtain data for the picture
    p1, p21, p22, p3, p4, p51, p52, ncu, s, e, p61, p62 = user_induction_simulation(beta=1, gamma=0.4, eta=1, model_dimension=15, final_recommend_num=100,
                                                                                    dataset='movielens 1m', method='kernel', improved=True, replace_num=10)

    name = '2'

    plt.figure(1, figsize=(22, 15), dpi=200)
    plt.rcParams.update({'font.size': 20})

    plt.subplot(2, 2, 1)
    plt.plot(p1)
    plt.title('User Feature Norm')
    plt.xlabel('Step')
    plt.ylabel('Norm')

    plt.subplot(2, 2, 2)
    plt.plot(p3)
    plt.title('Prob. of Well-correlation Items')
    plt.xlabel('Step')
    plt.ylabel('Prob. of System Select')

    plt.subplot(2, 2, 3)
    plt.plot(p21)
    plt.title('Likeable Item Prob.')
    plt.xlabel('Step')
    plt.ylabel('Prob. of user liking')

    plt.subplot(2, 2, 4)
    plt.plot(p22)
    plt.title('Non-likeable Item Prob.')
    plt.xlabel('Step')
    plt.ylabel('Prob. of user not liking')

    plt.savefig('./pic/pa' + name, bbox_inches='tight')  # !!Change!!
    plt.show()

    plt.figure(2, figsize=(22, 6), dpi=200)

    plt.subplot(1, 3, 1)
    plt.plot(p4)
    plt.title('Prob. of Forged Items')
    plt.xlabel('Step')
    plt.ylabel('Prob. of System Select')

    plt.subplot(1, 3, 2)
    plt.plot(p51)
    plt.title('Likeable Forged Item Prob.')
    plt.xlabel('Step')
    plt.ylabel('Prob. of user liking')

    plt.subplot(1, 3, 3)
    plt.plot(p52)
    plt.title('Non-likeable Forged Item Prob.')
    plt.xlabel('Step')
    plt.ylabel('Prob. of user liking')

    plt.savefig('./pic/induce' + name, bbox_inches='tight')  # !!Change!!
    plt.show()

    if ncu:
        for i in range(len(ncu)):
            if e[i] == s[i]:
                e[i] = 499

        plt.figure(3, figsize=(15, 6), dpi=200)
        plt.subplot(1, 2, 1)
        for i in range(len(ncu)):
            plt.scatter(x=np.arange(s[i], e[i], 1), y=p61[i, s[i]:e[i]], s=1.1, marker='o', label=str(ncu[i]))
            plt.plot(np.arange(s[i], e[i], 1), p61[i, s[i]:e[i]])
        plt.title('Non-certain Users Like Item Num.', fontsize=14)
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('Num. of Liked Items', fontsize=14)
        plt.legend(loc='best')
        plt.xlim(0, 500)
        plt.ylim(10, 90)
        plt.subplot(1, 2, 2)
        for i in range(len(ncu)):
            plt.scatter(x=np.arange(s[i], e[i], 1), y=p62[i, s[i]:e[i]], s=1.1, marker='o', label=str(ncu[i]))
            plt.plot(np.arange(s[i], e[i], 1), p62[i, s[i]:e[i]])
        plt.title('Non-certain Users Dislike Item Num.', fontsize=14)
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('Num. of Disliked Items', fontsize=14)
        plt.legend(loc='best')
        plt.xlim(0, 500)
        plt.ylim(10, 90)
        plt.savefig('./pic/ncu'+name, bbox_inches='tight')  # !!Change!!
        plt.show()
