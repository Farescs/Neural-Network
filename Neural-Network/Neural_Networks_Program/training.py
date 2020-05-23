import numpy as np
import pandas as p

"""
This class trains and predicts values for neural networks using stochastic gradient descent and backpropagation.
The supported activation functions for the neural network are the sigmoid, relu and identity functions.
The supported cost function is the sum of squares error function.
"""
class training:

    """
    This function performs the relu function and retuns the outputed value of that function.
    :param
        activation: the given dot product of weights and neuron values plus the bias of the respective layer.
    :returns
        np.maximum(0, activation): the relu function's outputed value.
    """
    def relu(self, activation):
        return np.maximum(0, activation)


    """
    This function performs the relu function's dervative and retuns the outputed value of that function.
    :param
        activation: the given dot product of weights and neuron values plus the bias of the respective layer.
    :returns
        activation: the derivative of the relu function.
    """
    def relu_derivative(self, activation):
        if not isinstance(activation, float):
            for i in range(len(activation)):
                activation[i <= 0] = 0
        elif activation <= 0:
            return 0

        return activation


    """
    This function performs the sigmoid function and retuns the outputed value of that function.
    :param
        activation: the given dot product of weights and neuron values plus the bias of the respective layer.
    :returns
        1 / (1 + np.exp(-activation)): the sigmoid function's outputed value.
    """
    def sigmoid(self, activation):
        return 1 / (1 + np.exp(-activation))


    """
    This function performs the sigmoid function's dervative and retuns the outputed value of that function.
    :param
        activation: the given dot product of weights and neuron values plus the bias of the respective layer.
    :returns
        training.sigmoid(self, activation) * (1 - training.sigmoid(self, activation)): the derivative of the sigmoid
        function.
    """
    def sigmoid_derivative(self, activation):
        return training.sigmoid(self, activation) * (1 - training.sigmoid(self, activation))


    """
    This function propagates forward through a given network.
    :params
        network: the given network.
        training_examples: the current trainng example from whatever input given.
    :returns
        network: the same given network, but with edited weights and neuron values.
        outputs_to_be_activated_at_each_layer: the outputs at each layer that would be activated.
    """
    def feed_forward(self, network):

        values_to_be_activated_for_backpropagation = {}
        layer = 0
        # Forward propagation through the network.
        for layer in range(1, len(network) - 1):

            # Initialize the dot product of the weights and the neurons of each layer plus the bias of that layer.
            dot_product_of_neurons_and_weights_plus_bias = np.zeros(len(network[layer]['neurons']), dtype=float)

            # Loop through each neuron and calculate next layer's neuron's activation value.
            for neuron in range(len(network[layer]['neurons'])):
                dot_product_of_neurons_and_weights_plus_bias[neuron] = \
                    np.dot(network[layer - 1]['neurons'],
                           network[layer]['weights'][neuron]) + network[layer]['bias'][neuron]

            values_to_be_activated_for_backpropagation[layer] = dot_product_of_neurons_and_weights_plus_bias #aL

            network[layer]['neurons'] = training.forward_activation(self,
                                                                    network,
                                                                    layer,
                                                                    dot_product_of_neurons_and_weights_plus_bias)

        # Finishing output layer's neuron values.
        dot_product_for_output_layer = np.dot(network[-2]['neurons'],
                                              network[-1]['weights']) + network[-1]['bias'][0]

        values_to_be_activated_for_backpropagation[layer + 1] = dot_product_for_output_layer

        network[-1]['neurons'] = training.forward_activation(self, network, len(network)-1,dot_product_for_output_layer)

        # Reindex the values to be activated at each layer.
        outputs_to_be_activated_at_each_layer = {}
        for index in range(len(values_to_be_activated_for_backpropagation)):
            outputs_to_be_activated_at_each_layer[index] = values_to_be_activated_for_backpropagation[index + 1]

        return network, outputs_to_be_activated_at_each_layer


    """
    This function is a help function for forward propagation. It computes the dot product values and 
    then pass those through an activation function.
    :param
        network: the given network.
        layer: the current layer in the network.
        dot_product_of_neurons_and_weights_plus_bias: the passed dot product to be passed through an activation 
        function.
    :returns
        network[layer]['neurons']: the neurons of the current layer that was just calculated.
    """
    def forward_activation(self, network, layer, dot_product_of_neurons_and_weights_plus_bias):
        len_of_dot_product_of_neurons_and_weights_plus_bias = 0
        # Conditionals to handle activation functions at each layer.
        # If no function is specified, the identity is the default.
        if network[layer]['layer_type'] != "output":
            len_of_dot_product_of_neurons_and_weights_plus_bias = len(dot_product_of_neurons_and_weights_plus_bias)

        if network[layer]['activation'] == "sigmoid":
            for activation in range(len_of_dot_product_of_neurons_and_weights_plus_bias):
                dot_product_of_neurons_and_weights_plus_bias[activation] = \
                    training.sigmoid(self, dot_product_of_neurons_and_weights_plus_bias[activation])

            network[layer]['neurons'] = dot_product_of_neurons_and_weights_plus_bias

        elif network[layer]['activation'] == "relu":
            for activation in range(len_of_dot_product_of_neurons_and_weights_plus_bias):
                dot_product_of_neurons_and_weights_plus_bias[activation] = \
                    training.relu(self, dot_product_of_neurons_and_weights_plus_bias[activation])

            network[layer]['neurons'] = dot_product_of_neurons_and_weights_plus_bias

        else:
            network[layer]['neurons'] = dot_product_of_neurons_and_weights_plus_bias

        return network[layer]['neurons']


    """
    This function performs the backpropagation algorithm on a given neural network and labels.
    :params
        network: the given network after forward propagation.
        labels: training labels as to adjust values correctly.
        predictions: output values after forward propagation to evaluate cost and adjustments.
    :returns
        gradient: the partial derivative of the cost function with respeact to the weights at each layer.
    """
    def backpropagation(self, network, outputs_to_be_activated_at_each_layer, label, prediction):

        delta_k = 0
        delta_j_layer = 0
        delta_weights = {}
        delta_bias = {}
        delta_Es_weights = {}
        delta_Es_bias = {}

        for layer in reversed(range(1, len(network))):

            # Activation for sigmoid.
            if network[layer]['activation'] is "sigmoid":
                if network[layer]['layer_type'] is "output":
                    activation = outputs_to_be_activated_at_each_layer[len(outputs_to_be_activated_at_each_layer) - 1]
                    derivative_activation = training.sigmoid_derivative(self, activation)
                else:
                    derivative_activation = training.sigmoid_derivative(self,
                                                                        outputs_to_be_activated_at_each_layer[layer])

            # Activation for relu.
            elif network[layer]['activation'] is "relu":
                if network[layer]['layer_type'] is "output":
                    activation = outputs_to_be_activated_at_each_layer[len(outputs_to_be_activated_at_each_layer) - 1]
                    derivative_activation = training.relu_derivative(self, activation)
                else:
                    derivative_activation = training.relu_derivative(self, outputs_to_be_activated_at_each_layer[layer])

            # Activation is linear in this case.
            else:
                if network[layer]['layer_type'] is "output":
                    activation = outputs_to_be_activated_at_each_layer[len(outputs_to_be_activated_at_each_layer) - 1]
                    derivative_activation = activation
                else:
                    derivative_activation = outputs_to_be_activated_at_each_layer[layer]

            # Actual backpropagation starting with the output layer and then moving backwards.
            if network[layer]['layer_type'] is "output":
                error_from_output = training.sum_of_squares(self, prediction, label)
                delta_k = -derivative_activation * error_from_output['derivative']
                delta_weights[delta_j_layer] = delta_k
                delta_E_weights = delta_weights[delta_j_layer] * network[layer - 1]['neurons']

                delta_bias[delta_j_layer] = error_from_output['derivative'] * 0.01
                delta_E_bias = delta_bias[delta_j_layer]

            else:
                deltas = {}
                for weight_vector in range(len(network[layer]['weights'])):
                    deltas[weight_vector] = np.dot(network[layer]['weights'][weight_vector],
                                                   delta_weights[delta_j_layer - 1])

                curr_delta = p.DataFrame(deltas)
                deltas = curr_delta.values
                delta_weights[delta_j_layer] = derivative_activation * deltas
                delta_E_weights = delta_weights[delta_j_layer] * network[layer - 1]['neurons']

                delta_bias[delta_j_layer]  = delta_bias[delta_j_layer - 1] * network[layer]['bias'] * 0.01
                delta_E_bias = delta_bias[delta_j_layer]

            delta_j_layer += 1
            delta_Es_weights[layer] = delta_E_weights
            delta_Es_bias[layer] = delta_E_bias

        return delta_Es_weights, delta_Es_bias


    """
    This function performs stochastic gradient decent optamization on a given neural network.
    :params
        network: the given network after backpropagated.
        learning_rate: the given learning rate for the network.
        labels: the given training labels for value adjustments.
    :returns
        network: the given network with updated adjusted values.
    """
    def update_weights(self, network, learning_rate, delta_Es_weights, delta_Es_bias):

        weight_error_signal = {}
        bias_error_signal = {}
        for index in range(len(delta_Es_weights)):
            weight_error_signal[index] = delta_Es_weights[index + 1]
            bias_error_signal[index] = delta_Es_bias[index + 1]

        network[-1]['weights'] -= learning_rate * weight_error_signal[len(weight_error_signal) - 1]
        network[-1]['bias'] -= learning_rate * bias_error_signal[len(bias_error_signal) - 1]
        # Iterate through the network.
        for layer in reversed(range(1, len(network) - 1)):

            error_signal_to_array = p.DataFrame(weight_error_signal[layer])
            weight_error_signal[layer] = error_signal_to_array.values

            for weight_vector in range(len(network[layer]['weights'])):
                network[layer]['weights'][weight_vector] += learning_rate * weight_error_signal[layer][weight_vector]

            network[layer]['bias'] += learning_rate * bias_error_signal[layer]

        return network


    """
    This function computes and returns the Cost and its derivative.
    The is function uses the Squared Error Cost function -> (1/2)*sum(Y - Y_hat)^2
    :param
        Y_hat: Predictions(activations) from a last layer, the output layer
        Y: labels of data
    :returns:
        cost: The Squared Error Cost result
        cost_derivative: gradient of Cost w.r.t the Y_hat
    """
    def sum_of_squares(self, Y_hat, Y):
        m = 1
        if not isinstance(Y, float):
            m = len(Y)
        cost = (1 / m) * np.sum(np.square(Y - Y_hat))
        cost_derivative = -(Y - Y_hat)
        return {'cost': cost, 'derivative': cost_derivative}


    """
    This function trains a given network for a certain amount of epochs.
    :param
        network: the given network.
        epochs: the number of iterations to train the network.
        labels: the given training labels for supervised learning.
    :returns
        network: the given network after training.
    """
    def train_network(self, network, epochs, labels, learning_rate):

        # sets_to_train_on = self.K_fold(self, network['input'])

        cost_history = []
        accuracy_history = []
        cost_boud = 0.001

        for epoch in range(epochs):
            predictions = {}
            print(epoch)
            for example in range(0, len(network[0]['input'])):
                network[0]['neurons'] = network[0]['input'][example]
                network, vals = training.feed_forward(self, network)
                prediction = network[-1]['neurons']
                cost = training.sum_of_squares(self, prediction, labels[example])
                delta_Es_weights, delta_Es_bias = training.backpropagation(self, network, vals,
                                                                           prediction, labels[example])
                network = training.update_weights(self, network, learning_rate, delta_Es_weights, delta_Es_bias)
                predictions[example] = prediction
                #if epoch is 0 or epoch is epochs - 1:
                    #print(f'pred {prediction}, label {labels[example]}, {example}, {cost["derivative"]}')
            ps = p.DataFrame(predictions, index=[0])
            predictions = ps.values
            d = training.sum_of_squares(self, predictions, labels)
            print(d['cost'])
        '''
        example = 0
        while example < epochs:#len(network[0]['input']):

            network[0]['neurons'] = network[0]['input'][example]
            network, vals = training.feed_forward(self, network)
            prediction = network[-1]['neurons']
            cost = training.sum_of_squares(self, prediction, labels[example])

            while not (-cost_boud < cost['derivative'] < cost_boud):

                network[0]['neurons'] = network[0]['input'][example]
                network, vals = training.feed_forward(self, network)
                prediction = network[-1]['neurons']
                cost = training.sum_of_squares(self, prediction, labels[example])
                delta_Es_weights, delta_Es_bias = training.backpropagation(self, network, vals, 
                                                                           prediction, labels[example])
                network = training.update_weights(self, network, learning_rate, delta_Es_weights, delta_Es_bias)
                print(f'pred {prediction}, label {labels[example]}')

            print(f'pred {prediction}, label {labels[example]}, {example}, {cost["derivative"]}')
            predictions[example] = prediction
            example += 1
        '''

        return network, predictions


'''
[-0.22022887 -0.23031511 -0.17526252 -0.249626   -0.26806215 -0.24707198
 -0.18773878 -0.22674582]
 
 [-0.25493835 -0.46358602 -0.228013   -0.40230705 -0.40737723 -0.29762333
 -0.4857131  -0.66353758]
 
 [-0.14722231 -0.81339843 -0.10553974 -0.45552393 -0.50507354 -0.19094691
 -0.28255413 -0.52401095]
 
 [-0.02868163 -0.24282625 -0.03333711 -0.1817565  -0.11807872 -0.0305139
 -0.08835918 -0.18548477]
'''