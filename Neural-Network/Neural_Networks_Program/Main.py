"""
Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town

 maybe this coule be dropped
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population

 MEDV     Median value of owner-occupied homes in $1000's

"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import Neural_Networks_Program.post_processing as post_processing
import Neural_Networks_Program.pre_processing as pre_processing
from Neural_Networks_Program.training import training


# This is the main driver class that wil run the program.
class Main:

    def __init__(self):
        # Get the data.
        data_frame = read_csv("training_data", delim_whitespace=True, header=None)
        data_set = data_frame.values

        sets = data_frame.values
        labs = sets[:, 13]
        #sets, m, s = pre_processing.pre_processing.normalize(self, sets)
        #print((sets * s) + m)

        # Normalize the dataset and split training data and labels into their own sets.
        normalised_data_set, mean, stnd = pre_processing.pre_processing.normalize(self, data_set)
        normalised_training_data = normalised_data_set[:, 0:13]
        normalised_labels = normalised_data_set[:, 13]

        # Perform PCA on normalised data.
        #normalised_pca_data = pre_processing.pre_processing.PCA(self, normalised_data, 13)

        """ 
        Initialize the network with the correct weights and also the input.
        Here is the architechture for the network.  There are 3 hidden layers, each using the sigmoid activation function 
        with an output layer use the relu activation function.  There are the respective input and output dimensions needed
        for each layer.
        """
        network = [
            {'layer_type': "input", 'neurons': np.zeros(13), 'activation': "sigmoid", 'input': normalised_training_data},

            {'layer_type': "hidden", 'neurons': np.zeros(13), 'activation': "sigmoid", 'bias': 1, 'weights': 0},

            {'layer_type': "output", 'neurons': np.zeros(1), 'activation': "sigmoid", 'bias': 1, 'weights': 0}
        ]

        for layer in range(len(network)):
            weight_vector_at_layer = {}
            for i in range(len(network[layer]['neurons'])):
                weight_vector_at_layer[i] = np.random.normal(0, 1, len(network[layer]['neurons']))
            network[layer]['weights'] = weight_vector_at_layer
            network[layer]['bias'] = np.random.normal(0, 1, len(network[layer]['neurons'])) * 0.01

        network[-1]['bias'] = np.random.normal(0, 1, len(network[len(network) - 1]['neurons'])) * 0.01
        network[-1]['weights'] = np.random.normal(0, 1, len(network[len(network) - 2]['neurons']))

        # sets_of_trainng_data = training.training.k_fold(self, training_data, 10)

        # Begin training the network with 10000 epochs.
        network, training_predictions = training.train_network(self, network, 100, normalised_labels, 0.1)

        ar = (training_predictions * stnd) + mean[-1]
        for i in range(len(ar[0])):
            print(f'prediction {ar[0][i]}, label {labs[i]}')

        # Get validation set.
        testing_data = read_csv("prediction data", delim_whitespace=True, header=None)
        test_data = testing_data.values

        # Normalize prediction set.
        normalised_test_data, mean_testing, stnd_testing = pre_processing.pre_processing.normalize(self, test_data)
        network[0]['input'] = normalised_test_data

        # Make pridictions on the actual prices against what the nn is giving.
        predictions = np.zeros(len(normalised_test_data))
        for example in range(len(normalised_test_data)):
            network[0]['neurons'] = network[0]['input'][example]
            network, vals = training.feed_forward(self, network)
            predictions[example] = network[-1]['neurons']

        print(predictions)
        # Post processing to make the output data readable and print those prices.
        print(post_processing.post_processing.post_process(self, predictions, mean, stnd))
        y = np.random.random(len(normalised_labels))
        plt.scatter(normalised_labels, y, c="red")
        plt.scatter(training_predictions, y, c="blue")
        # plt.scatter(predictions, y, c="green")
        plt.show()

if __name__ == '__main__':
    Main()
