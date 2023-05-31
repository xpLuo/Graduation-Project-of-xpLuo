from keras.models import Model
from keras.layers import Input, Embedding, Dot, add, Flatten, Lambda
from Utils import sigmoid


def build_network(num_user, num_item, latent, if_sigmoid=False):
    """
    Introduction:
        Another version of matrix factorization that uses neural network with embedding layers to represent the feature vector
        The input user id and item id has been resorted so that the neural network can have better performance
        This method is likely to generate very different feature vector compared with the kernel method
    Args:
        num_user (int): the number of users
        num_item (int): the number of items
        latent (int): how many latent factors to be included in one embedding layer, i.e. feature vector
        if_sigmoid (bool): whether to include a sigmoid layer at the end
    Returns:
        model (keras model): the network of the matrix factorization model
    """
    user_id = Input([1], name='user_id')
    item_id = Input([1], name='cust_id')
    user_embedding = Embedding(input_dim=num_user,
                               output_dim=latent,
                               name='user_embedding')(user_id)
    item_embedding = Embedding(input_dim=num_item,
                               output_dim=latent,
                               name='item_embedding')(item_id)
    user_bias = Embedding(input_dim=num_user,
                          output_dim=1,
                          name='user_bias')(user_id)
    item_bias = Embedding(input_dim=num_item,
                          output_dim=1,
                          name='item_bias')(item_id)

    output1 = Dot(2, name='inner_product')([user_embedding, item_embedding])
    output2 = add([output1, user_bias, item_bias])
    output3 = Flatten()(output2)

    if if_sigmoid:
        output4 = Lambda(sigmoid, name='sigmoid')(output3)
        model = Model(inputs=[user_id, item_id],
                      outputs=output4)
        model.compile(optimizer='adam', loss='binary_crossentropy')
    else:
        model = Model(inputs=[user_id, item_id],
                      outputs=output3)
        model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    return model
